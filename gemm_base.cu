// ============================================================================
// Stage 1: gemm_base — 最朴素版本
//   128 thread/CTA (4 warps, 2x2 布局), m16n8k16, fp16 累加
//   CTA tile = 128x128, K-tile BK=32, 与最终版 v12 的 tiling 完全一致
//
// 本阶段刻意"什么都不做"，作为演进的起点，三个最明显的问题：
//   1) 不使用 multistage : 每个 K-tile 都是
//      "cp.async 装载 -> wait -> __syncthreads -> 计算 -> __syncthreads"
//      串行流水，global->shared 的拷贝延迟完全暴露，无法与计算重叠。
//   2) 不解决 bank conflict: shared 直接行优先排布，无 swizzle。
//      行宽 64B = 16 word，行 r 与 r+2 落同一 16-bank 半区，
//      ldmatrix 读 8 行时 0&4、2&6 撞 bank，2-way 冲突。
//   3) 不做写回的合并访存: epilogue 直接按 mma fragment 布局逐个
//      写 2 个 fp16 (4B)，同一 warp 内相邻线程写相邻但错位的 4B,
//      全局写回不合并，浪费 128B 事务。
//
// 校验数据: A[i][k]=2^(i%4)∈{1,2,4,8}, B[k][n]=1 => C[i][n]=4096*2^(i%4)
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_fp16.h>

#define M 4096
#define N 4096
#define K 4096
#define BM 128   // CTA tile 行
#define BN 128   // CTA tile 列
#define BK 32    // K tile (2 次 m16n8k16), 与 v12 一致
#define NUM_KK (BK / 16)   // 2
#define CHUNKS (BK / 8)    // 4 个 16B chunk / 行

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src)
{
    unsigned s = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(s), "l"(gmem_src) : "memory");
}
__device__ __forceinline__ void cp_async_commit_group() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_async_wait_all()     { asm volatile("cp.async.wait_group 0;"); }

__global__ void gemm_amper(const half* __restrict__ gA,   // [M][K]
                           const half* __restrict__ gBT,  // [N][K] (B 转置)
                           half* __restrict__ gC)         // [M][N]
{
    __shared__ alignas(128) half sA[BM][BK];   // 128x32 = 8KB, 行 stride 64B (无 swizzle)
    __shared__ alignas(128) half sB[BN][BK];   // 8KB
    const int tid   = threadIdx.x;
    const int lane  = tid % 32;
    const int mwarp = (tid / 32) / 2;
    const int nwarp = (tid / 32) % 2;
    const int M0 = blockIdx.y * BM;
    const int N0 = blockIdx.x * BN;

    uint32_t C[4][8][2] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---------- 1) 单缓冲装载 ----------
        for (int i = tid; i < BM * CHUNKS; i += 128) {
            int row   = i / CHUNKS;
            int chunk = i % CHUNKS;
            cp_async_16(&sA[row][chunk * 8], &gA[(M0 + row) * K + k0 + chunk * 8]);
            cp_async_16(&sB[row][chunk * 8], &gBT[(N0 + row) * K + k0 + chunk * 8]);
        }
        cp_async_commit_group();
        cp_async_wait_all();   // 等全部拷贝完成 (延迟暴露)
        __syncthreads();

        // ---------- 2) 计算: BK=32 内 2 个 m16n8k16 ----------
        for (int kk = 0; kk < NUM_KK; ++kk) {
            const int kslice = kk * 16;
            const int grp    = lane >> 3;
            // A 片段: 4 个 8x8 象限, ldmatrix.x4
            uint32_t aFrag[4][4];
            const uint32_t baseA = (uint32_t)__cvta_generic_to_shared(&sA[0][0]);
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const int fr   = mwarp * 64 + mt * 16;
                const int mOff = (grp & 1) * 8;
                const int kOff = (grp >> 1) * 8;
                const int row  = fr + mOff + (lane & 7);
                // 无 swizzle: 物理列 = 逻辑列
                const uint32_t addr = baseA + (uint32_t)row * (BK * 2) + (uint32_t)(kslice + kOff) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(aFrag[mt][0]), "=r"(aFrag[mt][1]),
                               "=r"(aFrag[mt][2]), "=r"(aFrag[mt][3])
                             : "r"(addr));
            }
            // B 片段: 8 个 n-tile, 4 次 ldmatrix.x4
            uint32_t bFrag[8][2];
            const uint32_t baseB = (uint32_t)__cvta_generic_to_shared(&sB[0][0]);
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                const int fr   = nwarp * 64 + g * 16;
                const int mOff = (grp >> 1) * 8;
                const int kOff = (grp & 1) * 8;
                const int row  = fr + mOff + (lane & 7);
                const uint32_t addr = baseB + (uint32_t)row * (BK * 2) + (uint32_t)(kslice + kOff) * 2;
                uint32_t t0, t1, t2, t3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(t0), "=r"(t1), "=r"(t2), "=r"(t3)
                             : "r"(addr));
                bFrag[2 * g][0] = t0; bFrag[2 * g][1] = t1;
                bFrag[2 * g + 1][0] = t2; bFrag[2 * g + 1][1] = t3;
            }
            // mma 累加: 4x8 个 m16n8k16
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
        __syncthreads();   // 全部读完才允许下轮覆盖 sA/sB
    }

    // ---------- 3) epilogue: 朴素直接写回 (不合并) ----------
    // 每 reg 的 2 个 fp16 打包成 1 个 32b store; warp 内仍 8 行错位, 128B 事务浪费
    const int q = lane % 4;
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int row = M0 + mwarp * 64 + mt * 16 + lane / 4;
            const int col = N0 + nwarp * 64 + nt * 8 + 2 * q;
            half* p = &gC[row * N + col];
            // 寄存器 C[..][0] 低16b->p[0], 高16b->p[1], 整 32b 一次写 (小端天然对应)
            *(uint32_t*)p = C[mt][nt][0];             // 行内 4B
            *(uint32_t*)(p + 8 * N) = C[mt][nt][1];   // row+8 的 4B
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
            hA[i * K + k] = __float2half((float)(1 << (i & 3)));
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

    // 计时
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    for (int i = 0; i < 10; ++i) gemm_amper<<<grid, 128>>>(dA, dBT, dC);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int i = 0; i < 100; ++i) gemm_amper<<<grid, 128>>>(dA, dBT, dC);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    double tflops = 2.0 * M * N * K * 100 / (ms * 1e-3) / 1e12;
    printf("gemm_base: %.3f ms/iter  (%.1f TFLOP/s)\n", ms / 100, tflops);

    cudaMemcpy(hC, dC, szC * sizeof(half), cudaMemcpyDeviceToHost);

    size_t fail = 0;
    for (size_t i = 0; i < M; ++i) {
        const float exp = 4096.0f * (1 << (i & 3));
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
