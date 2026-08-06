// Ampere 单缓冲 GEMM 演示 (128 thread/CTA, 2x2 warp, m16n8k16, fp16 累加)
//
// 规格 (按需求):
//   - CTA = 128 threads (4 warps, 2x2x1 布局), CTA tile = 128x128x64, warp tile = 64x64
//   - K 方向循环 4 次: BK=64 的 K-tile 内拆成 4 次 m16n8k16 (kslice=0,16,32,48)
//     (全局 K=4096 -> 外层循环 4096/64 = 64 个 tile)
//   - 不用 multistage / double buffer: 每 tile 就是
//     "cp.async 装载 -> wait -> __syncthreads -> 计算 -> __syncthreads(防覆盖)"
//   - 不处理 bank conflict: shared 直接行优先排布
//   - 用 cp.async (global->shared) + ldmatrix.x4/x2 (shared->寄存器) + mma (寄存器->累加器)
//   - 累加精度 = FP16 (mma.sync.m16n8k16.row.col.f16.f16.f16.f16)
//
// 数据: A[i][k] = 2^(i%4) ∈ {1,2,4,8}, B[k][n] = 1
//       => C[i][n] = 4096 * 2^(i%4)
//       选 2 的幂: 每步累加量 16*2^(i%4) 都是该数量级 ulp 的整数倍,
//       fp16 全程精确, host 可做"逐元素精确比对".
//
// 校验: 4096x4096 全矩阵逐元素比对 host 期望值.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_fp16.h>

#define M 4096
#define N 4096
#define K 4096
#define BM 128   // CTA tile 行
#define BN 128   // CTA tile 列
#define BK 64    // K tile (四次 m16n8k16)

// 单条 cp.async 16B: 全局 src -> shared dst, 不阻塞发出线程
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

__global__ void gemm_amper(const half* __restrict__ gA,   // [M][K] 行优先
                           const half* __restrict__ gBT,  // [N][K] 行优先, 存 B 的转置 gBT[n][k]=B[k][n]
                           half* __restrict__ gC)         // [M][N] 行优先
{
    __shared__ alignas(128) half sA[BM][BK];   // A tile: 128 行(m) x 64 列(k) = 16KB
    __shared__ alignas(128) half sB[BN][BK];   // B^T tile: 128 行(n) x 64 列(k) = 16KB
    const int tid = threadIdx.x;
    const int lane   = tid % 32;
    const int mwarp  = (tid / 32) / 2;   // warp 2x2: M 方向 0..1
    const int nwarp  = (tid / 32) % 2;   //           N 方向 0..1
    const int M0 = blockIdx.y * BM;      // 本 CTA 的 M 起点
    const int N0 = blockIdx.x * BN;      // 本 CTA 的 N 起点

    // 累加器: 每 warp 64x64 = 4(m) x 8(n) 个 16x8 片段, 每个 2 个 u32, 初始 0
    uint32_t C[4][8][2] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---------- 1) cp.async 装载 A/B 当前 K-tile (单缓冲) ----------
        // A/B: 128行 x 128B = 16KB = 1024 个 16B chunk, 均分给 128 线程 (每线程 8 chunk)
        // (B 在全局已转置, 直接行优先拷, cp.async 不支持转置)
        for (int i = tid; i < BM * BK / 8; i += 128) {   // 1024 个 chunk
            int row  = i / 8;     // 0..127
            int chunk = i % 8;    // 行内 8 个 16B chunk (每 chunk = 8 fp16)
            cp_async_16(&sA[row][chunk * 8], &gA[(M0 + row) * K + k0 + chunk * 8]);
            cp_async_16(&sB[row][chunk * 8], &gBT[(N0 + row) * K + k0 + chunk * 8]);
        }
        cp_async_commit_group();
        cp_async_wait_all();   // 等"自己"的拷贝
        __syncthreads();       // 关键: 所有线程的拷贝对本 CTA 可见

        // ---------- 2) 计算: 本 BK=64 tile 内沿 K 循环 4 次 (四次 m16n8k16) ----------
        for (int kk = 0; kk < 4; ++kk) {
            const int kslice = kk * 16;   // shared 列偏移: 0/16/32/48

            // 装载 A 片段: 4 个 m-tile, ldmatrix.x4 得 4 个 u32
            //   layout: 4 个 8x8 象限 = (TL, BL, TR, BR), lane/8 决定象限
            //     lanes 0-7   -> TL (rows+0,   cols+0)
            //     lanes 8-15  -> BL (rows+8,   cols+0)
            //     lanes 16-23 -> TR (rows+0,   cols+8)
            //     lanes 24-31 -> BR (rows+8,   cols+8)
            uint32_t aFrag[4][4];
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const uint32_t base = (uint32_t)__cvta_generic_to_shared(&sA[0][0]);
                const int fr = mwarp * 64 + mt * 16;          // 片段首行 (shared 内)
                const int grp   = lane >> 3;                  // 0..3
                const int mOff  = (grp & 1) * 8;              // 0 或 8
                const int kOff  = (grp >> 1) * 8;             // 0 或 8
                const uint32_t addr = base + (uint32_t)(fr + mOff + (lane & 7)) * (BK * 2)
                                    + (uint32_t)(kslice + kOff) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(aFrag[mt][0]), "=r"(aFrag[mt][1]),
                               "=r"(aFrag[mt][2]), "=r"(aFrag[mt][3])
                             : "r"(addr));
            }
            // 装载 B 片段: 8 个 n-tile, ldmatrix.x2 得 2 个 u32
            //   两 8x8 矩阵 = (k=0..7, k=8..15) 同 8 行 n
            //     lanes 0-7  -> matrix 0 (kslice+0..7)
            //     lanes 8-15 -> matrix 1 (kslice+8..15)
            uint32_t bFrag[8][2];
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const uint32_t base = (uint32_t)__cvta_generic_to_shared(&sB[0][0]);
                const int fr = nwarp * 64 + nt * 8;           // 片段首行 (shared 内 = B 的 n)
                const int kOff = ((lane >> 3) & 1) * 8;       // lanes 0-7: 0, lanes 8-15: 8
                const uint32_t addr = base + (uint32_t)(fr + (lane & 7)) * (BK * 2)
                                    + (uint32_t)(kslice + kOff) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                             : "=r"(bFrag[nt][0]), "=r"(bFrag[nt][1]) : "r"(addr));
            }
            // mma 累加: 4x8 = 32 个 16x8 块, D = A*B + C, fp16 累加
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
        __syncthreads();   // 所有线程读完本 tile 才允许下轮 cp.async 覆盖 sA/sB
    }

    // ---------- 3) epilogue: 普通 store 写回全局 ----------
    // 线程 lane 持有 D[t/4][2q],[2q+1] (d0) 与 D[t/4+8][2q],[2q+1] (d1)
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
            // d1 是 D 的 8..15 行, 相对 d0 偏移 8 行 (不是 1 行!)
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
            hA[i * K + k] = __float2half((float)(1 << (i % 4)));   // 行 i: 1/2/4/8
    for (size_t n = 0; n < N; ++n)
        for (size_t k = 0; k < K; ++k)
            hBT[n * K + k] = __float2half(1.0f);                   // B[k][n] = 1

    half *dA, *dBT, *dC;
    cudaMalloc(&dA, szM * sizeof(half));
    cudaMalloc(&dBT, szN * sizeof(half));
    cudaMalloc(&dC, szC * sizeof(half));
    cudaMemcpy(dA, hA, szM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dBT, hBT, szN * sizeof(half), cudaMemcpyHostToDevice);

    dim3 grid(N / BN, M / BM);          // 32 x 32 = 1024 个 CTA
    gemm_amper<<<grid, 128>>>(dA, dBT, dC);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(hC, dC, szC * sizeof(half), cudaMemcpyDeviceToHost);

    // 逐元素校验: 期望 C[i][n] = 4096 * 2^(i%4)
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
