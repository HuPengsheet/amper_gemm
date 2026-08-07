// ============================================================================
// Stage 3: gemm_multistage_register — 多级流水 + 寄存器双缓冲 + swizzle
//   在 Stage 2 (gemm_epilogue) 基础上一次性补齐剩下三件事，这是最大的一次跳跃：
//
//   [A] cp.async 多级流水 (3-stage): global->shared 拷贝异步发出，提前
//       NUM_STAGES-1 个 K-tile，wait_group 延迟等待，拷贝延迟与当前 tile 的
//       mma 计算重叠。
//   [B] 寄存器双缓冲: 用两套 fragment 寄存器 (aBank[2]/bBank[2]) 交替，
//       计算当前 kk 的 mma 时同时把下一个 kk 的 fragment 预装载进另一套；
//       ldmatrix 的延迟被 mma 完全隐藏 (warp 内自掩盖，不依赖占用率)。
//       kk 循环显式展开，静态选择 CUR/NXT bank，避免运行时索引。
//   [C] swizzle: 此时 global/ldmatrix 延迟都藏住了，LDSM 的 bank conflict
//       才浮出来成为关键路径；用 (row>>1)&3 的 XOR swizzle 消除之。
//       swizzle 偏移在进主循环前一次算好 (precompute_ldsm_offsets)，
//       热循环里不再重算——不占寄存器，也不掉占用率。
//
//   这三点必须一起出现：单独 multistage 会因占用率减半 + ldmatrix 暴露而
//   更慢，单独 swizzle 在延迟未藏住时也不兑现。
//
// 两个次要结构优化: mma asm 非 volatile 让 ptxas 自由调度 HMMA 序列；
// cp.async 源指针按 BK 推进 (srcA[idx] += BK)，减少每 tile 地址计算。
//
// 最终版 gemm_finial = 本文件 + 占用率/流水细节调优。
// tiling 一致: BM=BN=128, BK=32。
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_fp16.h>

#define M 4096
#define N 4096
#define K 4096
#define BM 128
#define BN 128
#define BK 32
#define NUM_STAGES 3
#define NUM_KK (BK / 16)
#define CHUNKS (BK / 8)
#define CHUNKS_PER_THREAD (BM * CHUNKS / 128)
#define SWZ_MASK (CHUNKS - 1)

typedef uint32_t FragA[4][4];
typedef uint32_t FragB[8][2];

__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gmem_src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(smem_addr), "l"(gmem_src) : "memory");
}
__device__ __forceinline__ void cp_async_commit_group() { asm volatile("cp.async.commit_group;"); }

__device__ __forceinline__ int swizzled_col(int col_base, int row)
{
    return col_base ^ (((row >> 1) & SWZ_MASK) << 3);
}

// 发出一个 K-tile 的 global->shared 拷贝。源指针按 BK 推进 (v12 风格)。
__device__ __forceinline__ void issue_ktile_load(
    const half* (&srcA)[CHUNKS_PER_THREAD], const half* (&srcB)[CHUNKS_PER_THREAD],
    uint32_t sA_addr, uint32_t sB_addr, int tid)
{
    #pragma unroll
    for (int idx = 0; idx < CHUNKS_PER_THREAD; ++idx) {
        const int i     = tid + idx * 128;
        const int row   = i / CHUNKS;
        const int chunk = i % CHUNKS;
        const int swz   = swizzled_col(chunk * 8, row);
        cp_async_16(sA_addr + (uint32_t)(row * BK + swz) * 2, srcA[idx]);
        cp_async_16(sB_addr + (uint32_t)(row * BK + swz) * 2, srcB[idx]);
        srcA[idx] += BK;
        srcB[idx] += BK;
    }
    cp_async_commit_group();
}

// 预计算每个 kk 的 ldmatrix SMEM 偏移 (字节, 相对 stage base)
__device__ __forceinline__ void precompute_ldsm_offsets(
    int mwarp, int nwarp, int lane,
    uint32_t (&aLdsm)[4][4], uint32_t (&bLdsm)[4][4])
{
    const int grp   = lane >> 3;
    const int lane7 = lane & 7;
    const int swz   = ((lane7 >> 1) & SWZ_MASK) << 3;   // key = (row>>1)&3
    #pragma unroll
    for (int kk = 0; kk < NUM_KK; ++kk) {
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
    uint32_t sA_addr, uint32_t sB_addr, int kk,
    const uint32_t (&aLdsm)[4][4], const uint32_t (&bLdsm)[4][4],
    FragA& aFrag, FragB& bFrag)
{
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        const uint32_t addr = sA_addr + aLdsm[kk][mt];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(aFrag[mt][0]), "=r"(aFrag[mt][1]),
                       "=r"(aFrag[mt][2]), "=r"(aFrag[mt][3])
                     : "r"(addr));
    }
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        const uint32_t addr = sB_addr + bLdsm[kk][g];
        uint32_t t0, t1, t2, t3;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(t0), "=r"(t1), "=r"(t2), "=r"(t3)
                     : "r"(addr));
        bFrag[2 * g][0] = t0; bFrag[2 * g][1] = t1;
        bFrag[2 * g + 1][0] = t2; bFrag[2 * g + 1][1] = t3;
    }
}

__device__ __forceinline__ void mma_kk(
    const FragA& aFrag, const FragB& bFrag,
    uint32_t (&C)[4][8][2])
{
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt)
            asm(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
                : "=r"(C[mt][nt][0]), "=r"(C[mt][nt][1])
                : "r"(aFrag[mt][0]), "r"(aFrag[mt][1]),
                  "r"(aFrag[mt][2]), "r"(aFrag[mt][3]),
                  "r"(bFrag[nt][0]), "r"(bFrag[nt][1]),
                  "r"(C[mt][nt][0]), "r"(C[mt][nt][1]));
}

// 合并写回 epilogue
__device__ __forceinline__ void epilogue(
    half* sC, half* gC,
    const uint32_t (&C)[4][8][2],
    int mwarp, int nwarp, int lane, int tid,
    int M0, int N0)
{
    constexpr int SC_STRIDE = BN + 8;
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

__global__ void gemm_amper(const half* __restrict__ gA,
                                                     const half* __restrict__ gBT,
                                                     half* __restrict__ gC)
{
    extern __shared__ __align__(128) half smem_buf[];
    half* const sA_stages = smem_buf;
    half* const sB_stages = smem_buf + NUM_STAGES * BM * BK;

    const int tid   = threadIdx.x;
    const int lane  = tid % 32;
    const int mwarp = (tid / 32) / 2;
    const int nwarp = (tid / 32) % 2;
    const int M0 = blockIdx.y * BM;
    const int N0 = blockIdx.x * BN;
    const int NUM_K = K / BK;

    uint32_t C[4][8][2] = {};

    // 预计算 per-thread cp.async 源指针 (k0=0 base, 含 M0/N0 偏移)
    const half* srcA[CHUNKS_PER_THREAD];
    const half* srcB[CHUNKS_PER_THREAD];
    #pragma unroll
    for (int idx = 0; idx < CHUNKS_PER_THREAD; ++idx) {
        const int i     = tid + idx * 128;
        const int row   = i / CHUNKS;
        const int chunk = i % CHUNKS;
        srcA[idx] = &gA[(M0 + row) * K] + chunk * 8;
        srcB[idx] = &gBT[(N0 + row) * K] + chunk * 8;
    }

    // 预计算 ldmatrix 偏移
    uint32_t aLdsm[4][4], bLdsm[4][4];
    precompute_ldsm_offsets(mwarp, nwarp, lane, aLdsm, bLdsm);

    // Pipeline state
    int compute_stage = 0;
    int write_stage   = NUM_STAGES - 1;
    int k_tile_index  = NUM_STAGES - 1;

    const uint32_t sA0 = (uint32_t)__cvta_generic_to_shared(sA_stages);
    const uint32_t sB0 = (uint32_t)__cvta_generic_to_shared(sB_stages);
    constexpr uint32_t STAGE_BYTES = BM * BK * 2;

    // Prologue: issue 前 N-1 个 K-tile
    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; ++s) {
        issue_ktile_load(srcA, srcB, sA0 + s * STAGE_BYTES, sB0 + s * STAGE_BYTES, tid);
    }
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();

    // 预装载 kk=0 -> bank0
    FragA aBank[2];
    FragB bBank[2];
    load_kk_fragments(sA0, sB0, 0, aLdsm, bLdsm, aBank[0], bBank[0]);

    // ---------- 主循环: kk 显式展开, 静态双缓冲 ----------
    #define MM_STEP(kk, CUR, NXT)                                                          \
    do {                                                                                   \
        if (kk == NUM_KK - 1) {                                                            \
            compute_stage = (compute_stage + 1) % NUM_STAGES;                              \
            asm volatile("cp.async.wait_group 1;");                                        \
            __syncthreads();                                                               \
        }                                                                                  \
        {                                                                                  \
            const uint32_t sA = sA0 + compute_stage * STAGE_BYTES;                         \
            const uint32_t sB = sB0 + compute_stage * STAGE_BYTES;                         \
            load_kk_fragments(sA, sB, (kk + 1) & (NUM_KK - 1), aLdsm, bLdsm,               \
                              aBank[NXT], bBank[NXT]);                                     \
        }                                                                                  \
        if (kk == 0 && can_issue) {                                                        \
            issue_ktile_load(srcA, srcB,                                                   \
                             sA0 + write_stage * STAGE_BYTES,                              \
                             sB0 + write_stage * STAGE_BYTES, tid);                        \
        }                                                                                  \
        mma_kk(aBank[CUR], bBank[CUR], C);                                                 \
        if (kk == 0 && can_issue) {                                                        \
            k_tile_index++;                                                                \
            write_stage = compute_stage;                                                   \
        }                                                                                  \
    } while (0)

    for (int k_tile = 0; k_tile < NUM_K; ++k_tile) {
        const bool can_issue = (k_tile + NUM_STAGES - 1 < NUM_K);
        MM_STEP(0, 0, 1);
        MM_STEP(1, 1, 0);
    }
    #undef MM_STEP

    // ---------- 冲刷 + sync ----------
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    // ---------- epilogue ----------
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

    dim3 grid(N / BN, M / BM);
    gemm_amper<<<grid, 128, smem_size>>>(dA, dBT, dC);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    for (int i = 0; i < 10; ++i) gemm_amper<<<grid, 128, smem_size>>>(dA, dBT, dC);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int i = 0; i < 100; ++i) gemm_amper<<<grid, 128, smem_size>>>(dA, dBT, dC);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    double tflops = 2.0 * M * N * K * 100 / (ms * 1e-3) / 1e12;
    printf("gemm_multistage_register: %.3f ms/iter  (%.1f TFLOP/s)\n", ms / 100, tflops);

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
