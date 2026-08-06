// Ampere GEMM (m16n8k16, CTA tile 128x128x64, warp 2x2x1, fp16 累加, 3-stage cp.async pipeline)
//
// 严格对齐 cutedsl_gemm.py 的循环结构:
//   - 双层循环: 外层 k_tile, 内层 k_block (kk, NUM_KK=4, 全 unroll)
//   - 在 k_block == NUM_KK-1 (last) 时:
//       * advance compute_stage -> 指向下一个 k_tile 的 stage
//       * cp.async.wait_group(N-2)  <- 关键: wait 发生在 compute 末尾, 与最后一个 k_block 的 mma 重叠
//       * __syncthreads
//   - ldmatrix 加载 k_block_next 的 fragment (寄存器双缓冲, 与当前 mma 重叠)
//   - 在 k_block == 0 时:
//       * cp.async issue ktile k+N-1 -> write_stage
//       * commit_group, advance k_tile_index, rotate write_stage
//   - prologue 只 issue N-1 个 ktile, 第 N 个在 iter0 的 k_block=0 issue
//   - 进入主循环前 wait + preload k_block=0
//
// Epilogue: 128-bit 合并 store (reg -> SMEM -> reg -> GMEM)

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

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src)
{
    unsigned s = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :: "r"(s), "l"(gmem_src) : "memory");
}
__device__ __forceinline__ void cp_async_commit_group() { asm volatile("cp.async.commit_group;"); }

__device__ __forceinline__ int swizzled_col(int col_base, int row)
{
    return col_base ^ ((row & 7) << 3);
}

__device__ __forceinline__ void issue_ktile_load(
    const half* gA, const half* gBT,
    half* sA_stage, half* sB_stage,
    int M0, int N0, int k_tile, int tid)
{
    const int k0 = k_tile * BK;
    for (int i = tid; i < BM * BK / 8; i += 128) {
        int row   = i / 8;
        int chunk = i % 8;
        int phys_col = swizzled_col(chunk * 8, row);
        cp_async_16(&sA_stage[row * BK + phys_col], &gA[(M0 + row) * K + k0 + chunk * 8]);
        cp_async_16(&sB_stage[row * BK + phys_col], &gBT[(N0 + row) * K + k0 + chunk * 8]);
    }
    cp_async_commit_group();
}

__device__ __forceinline__ void load_kk_fragments(
    uint32_t sA_addr, uint32_t sB_addr,
    int mwarp, int nwarp, int lane, int kk,
    uint32_t aFrag[4][4], uint32_t bFrag[8][2])
{
    const int kslice = kk * 16;
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        const int fr = mwarp * 64 + mt * 16;
        const int grp  = lane >> 3;
        const int mOff = (grp & 1) * 8;
        const int kOff = (grp >> 1) * 8;
        const int row  = fr + mOff + (lane & 7);
        const int phys_col = swizzled_col(kslice + kOff, row);
        const uint32_t addr = sA_addr + (uint32_t)row * (BK * 2) + (uint32_t)phys_col * 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(aFrag[mt][0]), "=r"(aFrag[mt][1]),
                       "=r"(aFrag[mt][2]), "=r"(aFrag[mt][3])
                     : "r"(addr));
    }
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int fr = nwarp * 64 + nt * 8;
        const int kOff = ((lane >> 3) & 1) * 8;
        const int row  = fr + (lane & 7);
        const int phys_col = swizzled_col(kslice + kOff, row);
        const uint32_t addr = sB_addr + (uint32_t)row * (BK * 2) + (uint64_t)phys_col * 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                     : "=r"(bFrag[nt][0]), "=r"(bFrag[nt][1]) : "r"(addr));
    }
}

__device__ __forceinline__ void mma_kk(
    uint32_t aFrag[4][4], uint32_t bFrag[8][2],
    uint32_t (&C)[4][8][2])
{
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

__device__ __forceinline__ void epilogue(
    half* sC, half* gC,
    const uint32_t (&C)[4][8][2],
    int mwarp, int nwarp, int lane, int tid,
    int M0, int N0)
{
    // reg -> SMEM (按 MMA fragment layout)
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        const int row_base = mwarp * 64 + mt * 16 + lane / 4;
        const int row_hi   = row_base + 8;
        const int col_base_lane = nwarp * 64 + 2 * (lane % 4);
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int col = col_base_lane + nt * 8;
            *reinterpret_cast<uint32_t*>(&sC[row_base * BN + col]) = C[mt][nt][0];
            *reinterpret_cast<uint32_t*>(&sC[row_hi   * BN + col]) = C[mt][nt][1];
        }
    }
    __syncthreads();
    // SMEM -> reg -> GMEM (128b)
    #pragma unroll
    for (int b = 0; b < BM / 8; ++b) {
        const int row_in_batch = tid / 16;
        const int col_chunk    = tid % 16;
        const int row = b * 8 + row_in_batch;
        const int col = col_chunk * 8;
        const uint4 data = *reinterpret_cast<const uint4*>(&sC[row * BN + col]);
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

    const int tid = threadIdx.x;
    const int lane  = tid % 32;
    const int mwarp = (tid / 32) / 2;
    const int nwarp = (tid / 32) % 2;
    const int M0 = blockIdx.y * BM;
    const int N0 = blockIdx.x * BN;
    const int NUM_K = K / BK;

    uint32_t C[4][8][2] = {};

    // Pipeline state (对齐 CuTeDSL)
    int compute_stage = 0;                    // tCsA_p: 当前 k_tile 的 stage
    int write_stage   = NUM_STAGES - 1;       // smem_pipe_write: 下一个 issue 的 stage
    int k_tile_index  = NUM_STAGES - 1;       // 下一个要 issue 的 ktile 编号

    // ---------- Prologue: issue 前 N-1 个 K-tile ----------
    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; ++s) {
        issue_ktile_load(gA, gBT,
                         sA_stages + s * BM * BK,
                         sB_stages + s * BM * BK,
                         M0, N0, s, tid);
    }

    // ---------- Wait + preload k_block=0 (对齐 CuTeDSL 350-362) ----------
    asm volatile("cp.async.wait_group 1;");   // N-2 = 1
    __syncthreads();

    uint32_t aFrag[2][4][4];
    uint32_t bFrag[2][8][2];
    int cur = 0;
    {
        const uint32_t sA_addr = (uint32_t)__cvta_generic_to_shared(sA_stages + compute_stage * BM * BK);
        const uint32_t sB_addr = (uint32_t)__cvta_generic_to_shared(sB_stages + compute_stage * BM * BK);
        load_kk_fragments(sA_addr, sB_addr, mwarp, nwarp, lane, 0, aFrag[cur], bFrag[cur]);
    }

    // ---------- 双层主循环 (对齐 CuTeDSL 364-413) ----------
    for (int k_tile = 0; k_tile < NUM_K; ++k_tile) {
        #pragma unroll
        for (int kk = 0; kk < NUM_KK; ++kk) {
            // (1) k_block == last: advance compute_stage, wait, sync
            //     此时前 NUM_KK-1 个 mma 已发出, wait 可与最后一个 mma 的完成重叠
            if (kk == NUM_KK - 1) {
                compute_stage = (compute_stage + 1) % NUM_STAGES;
                asm volatile("cp.async.wait_group 1;");
                __syncthreads();
            }

            // (2) ldmatrix k_block_next (寄存器双缓冲: 与下面 mma 重叠)
            const int kk_load = (kk + 1) % NUM_KK;
            {
                const uint32_t sA_addr = (uint32_t)__cvta_generic_to_shared(sA_stages + compute_stage * BM * BK);
                const uint32_t sB_addr = (uint32_t)__cvta_generic_to_shared(sB_stages + compute_stage * BM * BK);
                load_kk_fragments(sA_addr, sB_addr, mwarp, nwarp, lane, kk_load, aFrag[cur ^ 1], bFrag[cur ^ 1]);
            }

            // (3) k_block == 0: issue cp.async for ktile k+N-1
            const bool can_issue = (k_tile + NUM_STAGES - 1 < NUM_K);
            if (kk == 0 && can_issue) {
                issue_ktile_load(gA, gBT,
                                 sA_stages + write_stage * BM * BK,
                                 sB_stages + write_stage * BM * BK,
                                 M0, N0, k_tile_index, tid);
            }

            // (4) mma on current k_block
            mma_kk(aFrag[cur], bFrag[cur], C);

            // (5) k_block == 0: advance k_tile_index, rotate write_stage (= old compute_stage)
            if (kk == 0 && can_issue) {
                k_tile_index++;
                write_stage = compute_stage;
            }

            cur ^= 1;
        }
    }

    // ---------- Final wait + sync (对齐 CuTeDSL 415-416) ----------
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

    const size_t smem_size = NUM_STAGES * 2 * BM * BK * sizeof(half);
    cudaFuncSetAttribute(gemm_amper, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(gemm_amper, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    dim3 grid(N / BN, M / BM);
    gemm_amper<<<grid, 128, smem_size>>>(dA, dBT, dC);
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
