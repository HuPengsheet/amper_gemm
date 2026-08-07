// ============================================================================
// Stage 2: gemm_epilogue — 写回用 shared memory 做合并访存
//   在 Stage 1 (gemm_base) 基础上只改 epilogue。tiling、装载、计算完全不变:
//   单缓冲 + 无 swizzle + cp.async 装载 + mma 计算, 与 base 逐字节相同。
//   区别只有最后写回 C 的方式。
//
// Stage 1 的 epilogue 按 mma fragment 布局直接写全局:
//   每线程每次只写 2 个 fp16 (4B), warp 内 32 个线程写的是 16 个不同行上
//   错位的 4B, 全局 store 的 128B 事务被严重浪费 (8 个 4B 才能填满一个事务)。
//
// 合并写回 (两阶段 transpose):
//   [1] reg -> SMEM : 按 mma fragment 布局写进一个 128 列宽的 C 缓冲
//   [2] SMEM -> GMEM: __syncthreads 后, 每线程按 "行内连续 16B" 重读 C 缓冲,
//       用 uint4 (128-bit) 合并写回 gC。32 线程覆盖 512B 连续地址,
//       warp 的 store 恰好填满 4 个 128B 事务, 完全合并。
//
// 复用 A/B 的 16KB 静态窗口做 C 缓冲 (不额外申请 smem, 保持 4 块/SM):
//   C tile 128x128 = 32KB 放不进去, 故分两半:
//   pass 0: mwarp 0 的 C 行 [0,64)  写入窗口 -> sync -> 合并写回
//   pass 1: mwarp 1 的 C 行 [64,128) 写入窗口 -> sync -> 合并写回
//   (C 缓冲 stride = BN = 128, 无 padding; 128 是 16 的倍数, uint4 仍对齐)
//   寻址全部用 uint32 单位 (行 stride = 64 u32), 让编译器把偏移折成常数,
//   写阶段只留 1 个 base 指针, 读回只留 1 个 smem + 1 组 gmem 指针。
//
// ⚠️ 必须用 -maxrregcount 128 编译 (不加 __launch_bounds__):
//   smem 中转写回比 base 的直接 4B 写回多出 ~11 个寄存器 (写阶段 + 读回指针,
//   与主循环尾部的地址寄存器在边界上叠加), 不加约束 ptxas 自然分配 132 regs
//   -> 65536/(128x132) = 3.88 -> 掉到 3 块/SM, 占用率损失直接盖过写回收益
//   (实测 4.3ms > base 3.5ms, 反而变慢)。
//   用 -maxrregcount 128 把分配压到 123 regs (0 spill), 保持 4 块/SM,
//   合并写回才兑现为真实提速 (实测 ~3.3ms < base ~3.5ms)。
//
// 编译: $NVCC -arch=sm_89 -O3 -maxrregcount 128 gemm_epilogue.cu -o build_gemm_epilogue
// tiling 与最终版一致: BM=BN=128, BK=32。
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
#define NUM_KK (BK / 16)
#define CHUNKS (BK / 8)

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src)
{
    unsigned s = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(s), "l"(gmem_src) : "memory");
}
__device__ __forceinline__ void cp_async_commit_group() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_async_wait_all()     { asm volatile("cp.async.wait_group 0;"); }

// 合并写回 epilogue: reg -> SMEM(128 宽) -> GMEM(128-bit), 两段复用 16KB 窗口
//   全部用 uint32 单位寻址 (行 stride = 128 half = 64 u32), 让编译器把偏移折成常数,
//   写阶段只留 1 个 base 指针, 读回只留 1 个 smem + 2 个 gmem 指针, 寄存器占用最小化。
__device__ __forceinline__ void epilogue(
    half* sC, half* gC,
    const uint32_t (&C)[4][8][2],
    int mwarp, int nwarp, int lane, int tid,
    int M0, int N0)
{
    constexpr int U32_STRIDE = BN / 2;      // 128 half / 2 = 64 u32/行
    constexpr int ROWS_PER_PASS = BM / 2;   // 64 行/pass, 64*128*2B = 16KB
    uint32_t* sCu = reinterpret_cast<uint32_t*>(sC);

    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        // [1] 只有 mwarp==pass 的 warp 持有这些行的 fragment, 写入窗口物理行 [0,64)
        //     C[mt][nt][0/1] -> u32 地址 = r0*64 + nwarp*32 + (lane&3) + mt*1024 + nt*4 (+512)
        if (mwarp == pass) {
            const uint32_t base = (uint32_t)(lane >> 2) * U32_STRIDE
                                + (uint32_t)nwarp * 32 + (uint32_t)(lane & 3);
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt)
                #pragma unroll
                for (int nt = 0; nt < 8; ++nt) {
                    sCu[base + mt * 16 * U32_STRIDE + nt * 4]            = C[mt][nt][0];
                    sCu[base + mt * 16 * U32_STRIDE + nt * 4 + 8 * U32_STRIDE] = C[mt][nt][1];
                }
        }
        __syncthreads();
        // [2] 全部 128 线程: 每线程 8 个 uint4, 按行内 16B 连续合并写回 gC
        uint32_t* sptr = sCu + (tid >> 4) * U32_STRIDE + (tid & 15) * 4;
        half* gptr = gC + (M0 + pass * ROWS_PER_PASS + (tid >> 4)) * N + N0 + (tid & 15) * 8;
        // 强制每次只处理 1 个 uint4: 避免 8 个 LDS.128 批量驻留 32 regs 挤爆预算
        #pragma unroll 1
        for (int b = 0; b < ROWS_PER_PASS / 8; ++b) {
            const uint4 data = *reinterpret_cast<const uint4*>(sptr);
            *reinterpret_cast<uint4*>(gptr) = data;
            sptr += 8 * U32_STRIDE;
            gptr += 8 * N;
        }
        __syncthreads();   // pass 1 覆盖窗口前确保 pass 0 读完成
    }
}

__global__ void gemm_amper(const half* __restrict__ gA,
                           const half* __restrict__ gBT,
                           half* __restrict__ gC)
{
    // 16KB 静态窗口: sA / sB 各 8KB, epilogue 复用整个窗口做 C 缓冲
    __shared__ alignas(128) half smem_buf[BM * BK + BN * BK];
    half* const sA = smem_buf;
    half* const sB = smem_buf + BM * BK;
    half* const sC = smem_buf;   // epilogue: 64 行 x 128 列

    const int tid   = threadIdx.x;
    const int lane  = tid % 32;
    const int mwarp = (tid / 32) / 2;
    const int nwarp = (tid / 32) % 2;
    const int M0 = blockIdx.y * BM;
    const int N0 = blockIdx.x * BN;

    uint32_t C[4][8][2] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM * CHUNKS; i += 128) {
            int row   = i / CHUNKS;
            int chunk = i % CHUNKS;
            cp_async_16(&sA[row * BK + chunk * 8], &gA[(M0 + row) * K + k0 + chunk * 8]);
            cp_async_16(&sB[row * BK + chunk * 8], &gBT[(N0 + row) * K + k0 + chunk * 8]);
        }
        cp_async_commit_group();
        cp_async_wait_all();
        __syncthreads();

        for (int kk = 0; kk < NUM_KK; ++kk) {
            const int kslice = kk * 16;
            const int grp    = lane >> 3;
            uint32_t aFrag[4][4];
            const uint32_t baseA = (uint32_t)__cvta_generic_to_shared(sA);
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const int fr   = mwarp * 64 + mt * 16;
                const int mOff = (grp & 1) * 8;
                const int kOff = (grp >> 1) * 8;
                const int row  = fr + mOff + (lane & 7);
                const uint32_t addr = baseA + (uint32_t)row * (BK * 2) + (uint32_t)(kslice + kOff) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(aFrag[mt][0]), "=r"(aFrag[mt][1]),
                               "=r"(aFrag[mt][2]), "=r"(aFrag[mt][3])
                             : "r"(addr));
            }
            uint32_t bFrag[8][2];
            const uint32_t baseB = (uint32_t)__cvta_generic_to_shared(sB);
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

    // ---------- epilogue: 合并写回 ----------
    epilogue(sC, gC, C, mwarp, nwarp, lane, tid, M0, N0);
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
    printf("gemm_epilogue: %.3f ms/iter  (%.1f TFLOP/s)\n", ms / 100, tflops);

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
