// Ampere GEMM (m16n8k16, CTA tile 128x128x64, warp 2x2x1, fp16 累加, 3-stage cp.async pipeline)
//
// v7: 针对 profile8 与 cutedsl 的对比做内层循环瘦身
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

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src)
{
    unsigned s = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
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
    for (int i = tid; i < (BM * BK) >> 3; i += 128) {
        int row   = i >> 3;
        int chunk = i & 7;
        int phys_col = swizzled_col(chunk * 8, row);
        cp_async_16(&sA_stage[row * BK + phys_col], &gA[(M0 + row) * K + k0 + chunk * 8]);
        cp_async_16(&sB_stage[row * BK + phys_col], &gBT[(N0 + row) * K + k0 + chunk * 8]);
    }
    cp_async_commit_group();
}

__device__ __forceinline__ void load_kk_fragments(
    uint32_t sA_addr, uint32_t sB_addr,
    int mwarp, int nwarp, int lane, int kk,
    FragA& aFrag, FragB& bFrag)
{
    const int kslice = kk * 16;
    const int grp   = lane >> 3;
    const int lane7 = lane & 7;
    const int swz   = lane7 << 3;          // swizzle 常量, 与 row&7 等价 (fr, mOff 都是 8 的倍数)

    // A: 4 ldmatrix.x4, 输出 4 连续寄存器 = aFrag[mt][0..3], 无需重组
    //    (A 的 mOff/kOff 与 v6 保持一致, mma A-fragment 的线程映射已验证正确)
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        const int fr   = mwarp * 64 + mt * 16;
        const int mOff = (grp & 1) * 8;
        const int kOff = (grp >> 1) * 8;
        const int row  = fr + mOff + lane7;
        const uint32_t addr = sA_addr + (uint32_t)row * (BK * 2) + (uint32_t)((kslice + kOff) ^ swz) * 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(aFrag[mt][0]), "=r"(aFrag[mt][1]),
                       "=r"(aFrag[mt][2]), "=r"(aFrag[mt][3])
                     : "r"(addr));
    }

    // B: 4 ldmatrix.x4, 交换 mOff/kOff -> 输出 {t0,t1} = n0 的 (klo,khi),
    //    {t2,t3} = n1 的 (klo,khi), 直接是连续寄存器对, 免去交叉重组 MOV
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        const int fr   = nwarp * 64 + g * 16;
        const int mOff = (grp >> 1) * 8;   // 与 A 相反
        const int kOff = (grp & 1) * 8;
        const int row  = fr + mOff + lane7;
        const uint32_t addr = sB_addr + (uint32_t)row * (BK * 2) + (uint32_t)((kslice + kOff) ^ swz) * 2;
        uint32_t t0, t1, t2, t3;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(t0), "=r"(t1), "=r"(t2), "=r"(t3)
                     : "r"(addr));
        bFrag[2*g][0]     = t0;   // n0 klo
        bFrag[2*g][1]     = t1;   // n0 khi (与 t0 相邻)
        bFrag[2*g+1][0]   = t2;   // n1 klo
        bFrag[2*g+1][1]   = t3;   // n1 khi (与 t2 相邻)
    }
}

__device__ __forceinline__ void mma_kk(
    FragA& aFrag, FragB& bFrag,
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

    // Pipeline state
    int compute_stage = 0;
    int write_stage   = NUM_STAGES - 1;
    int k_tile_index  = NUM_STAGES - 1;

    // ---------- Prologue: issue 前 N-1 个 K-tile ----------
    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; ++s) {
        issue_ktile_load(gA, gBT,
                         sA_stages + s * BM * BK,
                         sB_stages + s * BM * BK,
                         M0, N0, s, tid);
    }

    // ---------- Wait + preload k_block=0 (静态 bank0) ----------
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();

    FragA aBank[2];
    FragB bBank[2];
    {
        const uint32_t sA_addr = (uint32_t)__cvta_generic_to_shared(sA_stages);
        const uint32_t sB_addr = (uint32_t)__cvta_generic_to_shared(sB_stages);
        load_kk_fragments(sA_addr, sB_addr, mwarp, nwarp, lane, 0, aBank[0], bBank[0]);
    }

    // ---------- 主循环: kk 显式展开, 静态双缓冲 ----------
    // bank 轮转: kk=0 消费 bank0 / 载入 bank1; kk=1 消费 bank1 / 载入 bank0; ...
#define MM_STEP(kk, CUR, NXT)                                                          \
    do {                                                                               \
        if (kk == NUM_KK - 1) {                                                        \
            compute_stage = (compute_stage + 1) % NUM_STAGES;                          \
            asm volatile("cp.async.wait_group 1;");                                    \
            __syncthreads();                                                           \
        }                                                                              \
        {                                                                              \
            const uint32_t sA = (uint32_t)__cvta_generic_to_shared(                    \
                sA_stages + compute_stage * BM * BK);                                  \
            const uint32_t sB = (uint32_t)__cvta_generic_to_shared(                    \
                sB_stages + compute_stage * BM * BK);                                  \
            load_kk_fragments(sA, sB, mwarp, nwarp, lane, (kk + 1) & (NUM_KK - 1),     \
                              aBank[NXT], bBank[NXT]);                                 \
        }                                                                              \
        if (kk == 0 && can_issue) {                                                    \
            issue_ktile_load(gA, gBT,                                                  \
                             sA_stages + write_stage * BM * BK,                        \
                             sB_stages + write_stage * BM * BK,                        \
                             M0, N0, k_tile_index, tid);                               \
        }                                                                              \
        mma_kk(aBank[CUR], bBank[CUR], C);                                             \
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
