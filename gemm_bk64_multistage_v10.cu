// Ampere GEMM (m16n8k16, CTA tile 128x128x64, warp 2x2x1, fp16 累加, 3-stage cp.async pipeline)
//
// v10: 在源码层把下一 kk 的 ldmatrix 穿插进当前 kk 的 HMMA 序列
//   v9 证明死 NOP 与 volatile 无关 (ptxas 在 sm_120 上给每条 HMMA 后插
//   @!UPT UIADD3 URZ,URZ,URZ,URZ 填坑)。cutedsl 的 SASS 里 HMMA 之间是
//   真实的 LDSM/地址计算, 从不背靠背, 没有死 NOP。
//   v10 在 mma_kk 循环里按 2 条 HMMA 插 1 条下一 kk 的 ldmatrix (volatile,
//   固定顺序), 给 ptxas 真实指令填 HMMA 间隙, 期望消灭死 NOP 并提前启动
//   LDSM->HMMA 的延迟隐藏。
//
// v9: 去掉 mma asm 的 volatile, 让 ptxas 自由调度 HMMA 序列
//   (实测对死 NOP 无影响, 保留非 volatile 亦可)
//
// v8: 针对 profile8 与 cutedsl 的对比做内层循环瘦身
//   v6 每 ktile 执行 ~504 条指令, 而 cutedsl 只有 ~240 条, 差在:
//     1) B fragment 的 ldmatrix.x4 输出 {t0,t1,t2,t3} 被交叉拆成
//        bFrag[2g]={t0,t2} / bFrag[2g+1]={t1,t3}, mma 的 B 操作数是
//        连续寄存器对 (klo,khi), 编译器只能用 MOV 重组 -> 每 ktile ~200 条 MOV。
//        这里交换 B 的 mOff/kOff, 让 ldmatrix.x4 输出 {t0,t1}=n0 的 (klo,khi),
//        {t2,t3}=n1 的 (klo,khi), 输出直接就是连续对, 免去重组。
//     2) 运行时 cur 双缓冲索引 + 循环内 can_issue 判断, 阻止干净的寄存器分配。
//        改为 kk 显式展开的静态双缓冲 (bank0/bank1), can_issue 每 ktile 只算一次。
//     3) __launch_bounds__(128,1) 放开寄存器上限(<=255), 给分配器余量(cutedsl 用 196)。
//     4) swizzle 常量 (lane&7)<<3 提前算好, 不在每条 ldmatrix 里重算。
//
// Epilogue: 128-bit 合并 store (reg -> SMEM -> reg -> GMEM), 与 v6 一致。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_fp16.h>

#define M 4096
#define N 4096
#define K 4096
#define BM 128
#define BN 128
#define BK 64
#define NUM_STAGES 3
#define NUM_KK 4    // BK / 16 = 4 个 k_block per K-tile

typedef uint32_t FragA[4][4];   // A fragment bank: [mt][4]
typedef uint32_t FragB[8][2];   // B fragment bank: [nt][2]

__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gmem_src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(smem_addr), "l"(gmem_src) : "memory");
}
__device__ __forceinline__ void cp_async_commit_group() { asm volatile("cp.async.commit_group;"); }

__device__ __forceinline__ int swizzled_col(int col_base, int row)
{
    return col_base ^ ((row & 7) << 3);
}

__device__ __forceinline__ void issue_ktile_load(
    const half* (&srcA)[8], const half* (&srcB)[8],
    uint32_t sA_addr, uint32_t sB_addr,
    int tid)
{
    // 8 chunk-pairs fully unrolled; each thread has a fixed (row, chunk) per idx.
    // Source pointers are advanced by BK (in fp16) per call, i.e. one K-tile.
    #pragma unroll
    for (int idx = 0; idx < 8; ++idx) {
        const int i      = tid + idx * 128;
        const int row    = i >> 3;
        const int chunk  = i & 7;
        const int swz    = swizzled_col(chunk * 8, row);
        cp_async_16(sA_addr + (uint32_t)(row * BK + swz) * 2, srcA[idx]);
        cp_async_16(sB_addr + (uint32_t)(row * BK + swz) * 2, srcB[idx]);
        srcA[idx] += BK;   // BK fp16 = BK*2 bytes per k-tile
        srcB[idx] += BK;
    }
    cp_async_commit_group();
}

// Precompute the per-lane ldmatrix SMEM offsets (bytes, relative to stage base)
// for all 4 k-blocks. aLdsm[kk][mt], bLdsm[kk][g].
__device__ __forceinline__ void precompute_ldsm_offsets(
    int mwarp, int nwarp, int lane,
    uint32_t (&aLdsm)[4][4], uint32_t (&bLdsm)[4][4])
{
    const int grp   = lane >> 3;
    const int lane7 = lane & 7;
    const int swz   = lane7 << 3;
    #pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        const int kslice = kk * 16;
        #pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            const int fr   = mwarp * 64 + mt * 16;
            const int mOff = (grp & 1) * 8;
            const int kOff = (grp >> 1) * 8;
            const int row  = fr + mOff + lane7;
            aLdsm[kk][mt] = (uint32_t)row * (BK * 2) + (uint32_t)((kslice + kOff) ^ swz) * 2;
        }
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const int fr   = nwarp * 64 + g * 16;
            const int mOff = (grp >> 1) * 8;
            const int kOff = (grp & 1) * 8;
            const int row  = fr + mOff + lane7;
            bLdsm[kk][g]   = (uint32_t)row * (BK * 2) + (uint32_t)((kslice + kOff) ^ swz) * 2;
        }
    }
}

__device__ __forceinline__ void load_kk_fragments(
    uint32_t sA_addr, uint32_t sB_addr,
    int kk,
    const uint32_t (&aLdsm)[4][4], const uint32_t (&bLdsm)[4][4],
    FragA& aFrag, FragB& bFrag)
{
    // A: 4 ldmatrix.x4
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        const uint32_t addr = sA_addr + aLdsm[kk][mt];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(aFrag[mt][0]), "=r"(aFrag[mt][1]),
                       "=r"(aFrag[mt][2]), "=r"(aFrag[mt][3])
                     : "r"(addr));
    }
    // B: 4 ldmatrix.x4
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        const uint32_t addr = sB_addr + bLdsm[kk][g];
        uint32_t t0, t1, t2, t3;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(t0), "=r"(t1), "=r"(t2), "=r"(t3)
                     : "r"(addr));
        bFrag[2*g][0]   = t0;
        bFrag[2*g][1]   = t1;
        bFrag[2*g+1][0] = t2;
        bFrag[2*g+1][1] = t3;
    }
}

__device__ __forceinline__ void mma_kk_interleaved(
    FragA& aCur, FragB& bCur,          // 当前 kk 的 fragment (被 HMMA 读)
    FragA& aNxt, FragB& bNxt,          // 下一 kk 的 fragment (被 ldmatrix 写)
    uint32_t sA_addr, uint32_t sB_addr, int kk_load,
    const uint32_t (&aLdsm)[4][4], const uint32_t (&bLdsm)[4][4],
    uint32_t (&C)[4][8][2])
{
    // 16 条 HMMA (mt=0..3, nt=0..7) 中每 2 条穿插 1 条下一 kk 的 ldmatrix:
    //   前 4 条穿插 A 的 4 个 mt (mt=0..3), 之后穿插 B 的 4 个 g (g=0..3)。
    // 顺序固定 (volatile), 与 load_kk_fragments 的语义完全一致。
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
                : "=r"(C[mt][nt][0]), "=r"(C[mt][nt][1])
                : "r"(aCur[mt][0]), "r"(aCur[mt][1]),
                  "r"(aCur[mt][2]), "r"(aCur[mt][3]),
                  "r"(bCur[nt][0]), "r"(bCur[nt][1]),
                  "r"(C[mt][nt][0]), "r"(C[mt][nt][1]));
            // 每 2 条 HMMA 后穿插 1 条 ldmatrix (mt*8+nt 为 1,3,5,...)
            if (((mt * 8 + nt) & 1) == 1) {
                const int ld = (mt * 8 + nt) >> 1;   // 0..7
                if (ld < 4) {
                    const uint32_t addr = sA_addr + aLdsm[kk_load][ld];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(aNxt[ld][0]), "=r"(aNxt[ld][1]),
                                   "=r"(aNxt[ld][2]), "=r"(aNxt[ld][3])
                                 : "r"(addr));
                } else {
                    const uint32_t addr = sB_addr + bLdsm[kk_load][ld - 4];
                    uint32_t t0, t1, t2, t3;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(t0), "=r"(t1), "=r"(t2), "=r"(t3)
                                 : "r"(addr));
                    const int g = ld - 4;
                    bNxt[2*g][0]   = t0;
                    bNxt[2*g][1]   = t1;
                    bNxt[2*g+1][0] = t2;
                    bNxt[2*g+1][1] = t3;
                }
            }
        }
    }
}

__device__ __forceinline__ void epilogue(
    half* sC, half* gC,
    const uint32_t (&C)[4][8][2],
    int mwarp, int nwarp, int lane, int tid,
    int M0, int N0)
{
    // sC 用 padded stride: SC_STRIDE = BN+8 fp16 = 272B = 68 bank
    // 相邻行 bank 偏移 4, 32 lane 均匀分布到 32 bank -> 无冲突
    // 且 272 是 16 的倍数 -> uint4 (128b) 访问对齐
    constexpr int SC_STRIDE = BN + 8;
    // reg -> SMEM (按 MMA fragment layout, padded stride)
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        const int row_base = mwarp * 64 + mt * 16 + (lane >> 2);
        const int row_hi   = row_base + 8;
        const int col_base_lane = nwarp * 64 + 2 * (lane & 3);
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int col = col_base_lane + nt * 8;
            *reinterpret_cast<uint32_t*>(&sC[row_base * SC_STRIDE + col]) = C[mt][nt][0];
            *reinterpret_cast<uint32_t*>(&sC[row_hi   * SC_STRIDE + col]) = C[mt][nt][1];
        }
    }
    __syncthreads();
    // SMEM -> reg -> GMEM (128b, padded stride 读)
    #pragma unroll
    for (int b = 0; b < BM >> 3; ++b) {
        const int row_in_batch = tid >> 4;
        const int col_chunk    = tid & 15;
        const int row = b * 8 + row_in_batch;
        const int col = col_chunk * 8;
        const uint4 data = *reinterpret_cast<const uint4*>(&sC[row * SC_STRIDE + col]);
        *reinterpret_cast<uint4*>(&gC[(M0 + row) * N + N0 + col]) = data;
    }
}

__global__ void __launch_bounds__(128, 1)
gemm_amper(const half* __restrict__ gA,
           const half* __restrict__ gBT,
           half* __restrict__ gC)
{
    extern __shared__ __align__(128) half smem_buf[];
    half* const sA_stages = smem_buf;
    half* const sB_stages = smem_buf + NUM_STAGES * BM * BK;

    // ---------- Grid rasterization (与 v6/cutedsl 一致) ----------
    constexpr int RASTER_F = 8;
    const int M_tile = blockIdx.x >> 3;
    const int N_tile = (blockIdx.x & (RASTER_F - 1)) + blockIdx.y * RASTER_F;
    if (M_tile >= M / BM || N_tile >= N / BN) return;
    const int M0 = M_tile * BM;
    const int N0 = N_tile * BN;

    const int tid = threadIdx.x;
    const int lane  = tid & 31;
    const int mwarp = tid >> 6;
    const int nwarp = (tid >> 5) & 1;
    const int NUM_K = K / BK;

    uint32_t C[4][8][2] = {};

    // ---------- Precompute per-thread cp.async source pointers (k0=0 base) ----------
    // 8 chunks per thread; each advances by BK fp16 per issued k-tile.
    const half* srcA[8];
    const half* srcB[8];
    #pragma unroll
    for (int idx = 0; idx < 8; ++idx) {
        const int i     = tid + idx * 128;
        const int row   = i >> 3;
        const int chunk = i & 7;
        srcA[idx] = &gA[(M0 + row) * K] + chunk * 8;
        srcB[idx] = &gBT[(N0 + row) * K] + chunk * 8;
    }

    // ---------- Precompute ldmatrix SMEM offsets per k-block ----------
    uint32_t aLdsm[4][4], bLdsm[4][4];
    precompute_ldsm_offsets(mwarp, nwarp, lane, aLdsm, bLdsm);

    // ---------- Pipeline state ----------
    int compute_stage = 0;
    int write_stage   = NUM_STAGES - 1;
    int k_tile_index  = NUM_STAGES - 1;

    // Stage SMEM byte addresses (offset within shared window), precomputed.
    const uint32_t sA0 = (uint32_t)__cvta_generic_to_shared(sA_stages);
    const uint32_t sB0 = (uint32_t)__cvta_generic_to_shared(sB_stages);
    constexpr uint32_t STAGE_BYTES = BM * BK * 2;   // 16KB

    // ---------- Prologue: issue 前 N-1 个 K-tile ----------
    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; ++s) {
        issue_ktile_load(srcA, srcB,
                         sA0 + s * STAGE_BYTES, sB0 + s * STAGE_BYTES, tid);
    }

    // ---------- Wait + preload k_block=0 (静态 bank0) ----------
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();

    FragA aBank[2];
    FragB bBank[2];
    {
        load_kk_fragments(sA0 + compute_stage * STAGE_BYTES, sB0 + compute_stage * STAGE_BYTES,
                          0, aLdsm, bLdsm, aBank[0], bBank[0]);
    }

    // ---------- 主循环: kk 显式展开, 静态双缓冲 ----------
    // 注意: issue 后源指针已推进, 与 k_tile_index 同步。
#define MM_STEP(kk, CUR, NXT)                                                          \
    do {                                                                               \
        if (kk == NUM_KK - 1) {                                                        \
            compute_stage = (compute_stage + 1) % NUM_STAGES;                          \
            asm volatile("cp.async.wait_group 1;");                                    \
            __syncthreads();                                                           \
        }                                                                              \
        if (kk == 0 && can_issue) {                                                    \
            issue_ktile_load(srcA, srcB,                                               \
                             sA0 + write_stage * STAGE_BYTES,                          \
                             sB0 + write_stage * STAGE_BYTES, tid);                    \
        }                                                                              \
        {                                                                              \
            const uint32_t sA = sA0 + compute_stage * STAGE_BYTES;                     \
            const uint32_t sB = sB0 + compute_stage * STAGE_BYTES;                     \
            mma_kk_interleaved(aBank[CUR], bBank[CUR], aBank[NXT], bBank[NXT],         \
                               sA, sB, (kk + 1) & (NUM_KK - 1), aLdsm, bLdsm, C);      \
        }                                                                              \
        if (kk == 0 && can_issue) {                                                    \
            k_tile_index++;                                                            \
            write_stage = compute_stage;                                               \
        }                                                                              \
    } while (0)

    for (int k_tile = 0; k_tile < NUM_K; ++k_tile) {
        const bool can_issue = (k_tile + NUM_STAGES - 1 < NUM_K);
        MM_STEP(0, 0, 1);
        MM_STEP(1, 1, 0);
        MM_STEP(2, 0, 1);
        MM_STEP(3, 1, 0);
    }
#undef MM_STEP

#undef MM_STEP

    // ---------- Final wait + sync ----------
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    // ---------- Epilogue ----------
    epilogue(sA_stages, gC, C, mwarp, nwarp, lane, tid, M0, N0);
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

    const size_t smem_size = NUM_STAGES * 2 * BM * BK * sizeof(half);
    cudaFuncSetAttribute(gemm_amper, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(gemm_amper, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    constexpr int RASTER_F = 8;
    const int M_grid = M / BM;
    const int N_grid = N / BN;
    dim3 grid(M_grid * RASTER_F, (N_grid + RASTER_F - 1) / RASTER_F);
    gemm_amper<<<grid, 128, smem_size>>>(dA, dBT, dC);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    // 计时: 预热 + 100 次
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    for (int i = 0; i < 10; ++i) gemm_amper<<<grid, 128, smem_size>>>(dA, dBT, dC);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int i = 0; i < 100; ++i) gemm_amper<<<grid, 128, smem_size>>>(dA, dBT, dC);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    double tflops = 2.0 * M * N * K * 100 / (ms * 1e-3) / 1e12;
    printf("v7 kernel: %.3f ms/iter  (%.1f TFLOP/s)\n", ms / 100, tflops);

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
