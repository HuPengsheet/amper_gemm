// Ampere 单缓冲 GEMM (m16n8k16, CTA tile 128x128x64, warp 2x2x1, fp16 累加)
//
// 与 gemm_base.cu 区别: 用 XOR swizzle (无 padding) 消除 shared memory bank conflict.
// 与 gemm_bk64_pad.cu 区别: 不浪费 shared, 行 stride 仍 = 128B; 通过 XOR 重映射 chunk 索引.
//
// Bank conflict 分析 (无 swizzle, stride=128B=32 bank):
//   相邻行同列落同 bank -> ldmatrix.x4 8 行命中同 4 bank -> 8-way 冲突 (8 cycle).
//
// Swizzle:
//   physical_col = logical_col XOR ((row & 7) << 3)
//   等价于 chunk 级重映射: physical_chunk = logical_chunk XOR (row & 7)
//   (chunk = 8 fp16 = 16B, 一行 64 fp16 = 8 chunk)
//
//   - XOR 只动 bit 3-5, 8 fp16 的逻辑连续列 -> 物理 8 连续列 (ldmatrix 16B 读不变语义)
//   - 行 r 的 chunk c 存到物理 chunk (c XOR r), 一行内 8 chunk 仍是 8 个不同物理 chunk -> 不溢出
//
// 冲突统计 (ldmatrix.x4, 32 lane 同时):
//   每 bank 组(4 bank) 命中 4 lane -> 4-way -> 4 cycle = 带宽极限 (512B/128B/cycle).
//   即使理论上完全无冲突, 也不可能 < 4 cycle. 已达最优.
//
// 相对 padding 的优势:
//   - shared 用量 = 2*128*64*2 = 32KB (padding 版 36KB)
//   - 行 stride 仍是 128B (2 的幂, 对 alignment/cp.async 更友好)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_fp16.h>

#define M 4096
#define N 4096
#define K 4096
#define BM 128
#define BN 128
#define BK 64    // K tile (四次 m16n8k16), 无 padding

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src)
{
    unsigned s = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :: "r"(s), "l"(gmem_src) : "memory");
}
__device__ __forceinline__ void cp_async_commit_group()
{
    asm volatile("cp.async.commit_group;");
}
__device__ __forceinline__ void cp_async_wait_all()
{
    asm volatile("cp.async.wait_group 0;");
}

// Swizzle: physical_col = logical_col XOR ((row & 7) << 3)
// 对 8-fp16 对齐的 col_base (即 chunk*8 / kslice / kslice+kOff), 等价于 chunk 级 XOR.
__device__ __forceinline__ int swizzled_col(int col_base, int row)
{
    return col_base ^ ((row & 7) << 3);
}

__global__ void gemm_amper(const half* __restrict__ gA,
                           const half* __restrict__ gBT,
                           half* __restrict__ gC)
{
    __shared__ alignas(128) half sA[BM][BK];   // 128 x 64 = 16KB, 行 stride = 128B
    __shared__ alignas(128) half sB[BN][BK];   // 128 x 64 = 16KB
    const int tid = threadIdx.x;
    const int lane   = tid % 32;
    const int mwarp  = (tid / 32) / 2;
    const int nwarp  = (tid / 32) % 2;
    const int M0 = blockIdx.y * BM;
    const int N0 = blockIdx.x * BN;

    uint32_t C[4][8][2] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---------- 1) cp.async 装载 (写入用 swizzled 物理 chunk 地址) ----------
        // 逻辑 chunk c (= cols [c*8 .. c*8+7]) -> 物理 chunk (c XOR (row&7))
        for (int i = tid; i < BM * BK / 8; i += 128) {   // 1024 个 chunk
            int row   = i / 8;
            int chunk = i % 8;
            int phys_col = swizzled_col(chunk * 8, row);   // = (chunk ^ (row&7)) * 8
            cp_async_16(&sA[row][phys_col], &gA[(M0 + row) * K + k0 + chunk * 8]);
            cp_async_16(&sB[row][phys_col], &gBT[(N0 + row) * K + k0 + chunk * 8]);
        }
        cp_async_commit_group();
        cp_async_wait_all();
        __syncthreads();

        // ---------- 2) 计算: BK=64 内沿 K 循环 4 次 (m16n8k16) ----------
        for (int kk = 0; kk < 4; ++kk) {
            const int kslice = kk * 16;

            // A 片段: 4 m-tile, ldmatrix.x4 得 4 u32
            uint32_t aFrag[4][4];
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const uint32_t base = (uint32_t)__cvta_generic_to_shared(&sA[0][0]);
                const int fr  = mwarp * 64 + mt * 16;
                const int grp = lane >> 3;
                const int mOff = (grp & 1) * 8;
                const int kOff = (grp >> 1) * 8;
                const int row = fr + mOff + (lane & 7);
                const int phys_col = swizzled_col(kslice + kOff, row);
                const uint32_t addr = base + (uint32_t)row * (BK * 2)
                                    + (uint32_t)phys_col * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(aFrag[mt][0]), "=r"(aFrag[mt][1]),
                               "=r"(aFrag[mt][2]), "=r"(aFrag[mt][3])
                             : "r"(addr));
            }
            // B 片段: 8 n-tile, ldmatrix.x2 得 2 u32
            uint32_t bFrag[8][2];
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const uint32_t base = (uint32_t)__cvta_generic_to_shared(&sB[0][0]);
                const int fr = nwarp * 64 + nt * 8;
                const int kOff = ((lane >> 3) & 1) * 8;
                const int row = fr + (lane & 7);
                const int phys_col = swizzled_col(kslice + kOff, row);
                const uint32_t addr = base + (uint32_t)row * (BK * 2)
                                    + (uint32_t)phys_col * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                             : "=r"(bFrag[nt][0]), "=r"(bFrag[nt][1]) : "r"(addr));
            }
            // mma: m16n8k16, fp16 累加
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt)
                #pragma unroll
                for (int nt = 0; nt < 8; ++nt)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
                        : "=r"(C[mt][nt][0]), "=r"(C[mt][nt][1])
                        : "r"(aFrag[mt][0]), "r"(aFrag[mt][1]),
                          "r"(aFrag[mt][2]), "r"(aFrag[mt][3]),
                          "r"(bFrag[nt][0]), "r"(bFrag[nt][1]),
                          "r"(C[mt][nt][0]), "r"(C[mt][nt][1]));
        }
        __syncthreads();
    }

    // ---------- 3) epilogue: 与 base 版完全相同 ----------
    const int q = lane % 4;
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int row = M0 + mwarp * 64 + mt * 16 + lane / 4;
            const int col = N0 + nwarp * 64 + nt * 8 + 2 * q;
            half* p = &gC[row * N + col];
            p[0]   = __ushort_as_half((unsigned short)(C[mt][nt][0] & 0xFFFFu));
            p[1]   = __ushort_as_half((unsigned short)(C[mt][nt][0] >> 16));
            p[8*N]   = __ushort_as_half((unsigned short)(C[mt][nt][1] & 0xFFFFu));
            p[8*N+1] = __ushort_as_half((unsigned short)(C[mt][nt][1] >> 16));
        }
}

int main()
{
    const size_t szM = (size_t)M * K, szN = (size_t)N * K, szC = (size_t)M * N;

    half* hA  = (half*)malloc(szM * sizeof(half));
    half* hBT = (half*)malloc(szN * sizeof(half));
    half* hC  = (half*)malloc(szC * sizeof(half));
    if (!hA || !hBT || !hC) { fprintf(stderr, "host alloc fail\n"); return 1; }

    for (size_t i = 0; i < M; ++i)
        for (size_t k = 0; k < K; ++k)
            hA[i * K + k] = __float2half((float)(1 << (i % 4)));
    for (size_t n = 0; n < N; ++n)
        for (size_t k = 0; k < K; ++k)
            hBT[n * K + k] = __float2half(1.0f);

    half *dA, *dBT, *dC;
    cudaMalloc(&dA, szM * sizeof(half));
    cudaMalloc(&dBT, szN * sizeof(half));
    cudaMalloc(&dC, szC * sizeof(half));
    cudaMemcpy(dA, hA, szM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dBT, hBT, szN * sizeof(half), cudaMemcpyHostToDevice);

    dim3 grid(N / BN, M / BM);
    gemm_amper<<<grid, 128>>>(dA, dBT, dC);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(hC, dC, szC * sizeof(half), cudaMemcpyDeviceToHost);

    size_t fail = 0;
    for (size_t i = 0; i < M; ++i) {
        const float exp = 4096.0f * (1 << (i % 4));
        for (size_t n = 0; n < N; ++n) {
            if ((float)__half2float(hC[i * N + n]) != exp) fail++;
        }
    }
    printf("sample: C[0][0]=%.0f C[1][0]=%.0f C[2][0]=%.0f C[3][0]=%.0f\n",
           __half2float(hC[0]), __half2float(hC[1 * N]), __half2float(hC[2 * N]), __half2float(hC[3 * N]));
    printf("host check: %s  (mismatch %zu / %zu)\n",
           fail ? "FAIL" : "PASS", fail, (size_t)M * N);

    free(hA); free(hBT); free(hC);
    cudaFree(dA); cudaFree(dBT); cudaFree(dC);
    return 0;
}
