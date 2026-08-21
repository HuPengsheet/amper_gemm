// ============================================================================
// 实验: gemm_base_multistage — base 只加 multistage 流水
//   研究问题: 在 gemm_base 基础上单独引入 cp.async 多级流水, 能不能提速?
//
//   与 gemm_base 的对照:
//     - compute (ldmatrix + mma) 逐字节相同
//     - epilogue (朴素直接写回, 不合并) 逐字节相同
//     - 无 swizzle (smem 线性排布)
//     - 无寄存器双缓冲 (每个 kk 现场装载 fragment, 装完就算)
//     唯一区别: 装载从"单缓冲 + wait_all"换成"3-stage cp.async 流水",
//     提前 NUM_STAGES-1 个 K-tile 发出拷贝, wait_group 1 延迟等待,
//     让 global->shared 拷贝延迟与当前 tile 的 mma 计算重叠。
//
//   代价: smem 16KB -> 48KB (3 个 stage), 块/SM 由 4 掉到 2 (smem 限制)。
//   理论上流水藏住了 global 延迟, 但占用率减半 + ldmatrix 延迟裸露。
//   实测见下方 main 输出。
//
// 编译: $NVCC -arch=sm_89 -O3 gemm_base_multistage.cu -o build_gemm_base_multistage
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
#define NUM_STAGES 3
#define CHUNKS_PER_THREAD (BM * CHUNKS / 128)

__device__ __forceinline__ void cp_async_16(uint32_t smem_dst, const void* gmem_src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(smem_dst), "l"(gmem_src) : "memory");
}
__device__ __forceinline__ void cp_async_commit_group() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_async_wait_all()     { asm volatile("cp.async.wait_group 0;"); }
__device__ __forceinline__ void cp_async_wait_near()    { asm volatile("cp.async.wait_group 1;"); }

__global__ void gemm_amper(const half* __restrict__ gA,   // [M][K]
                           const half* __restrict__ gBT,  // [N][K] (B 转置)
                           half* __restrict__ gC)         // [M][N]
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

    // per-thread 全局源偏移 (32-bit half 元素偏移, 不含 k0).
    // 存 32-bit 偏移而非 64-bit 指针: 否则 prologue 预存指针那一步就要吃掉一套 LEA(64-bit 地址合成).
    int offA[CHUNKS_PER_THREAD];
    int offB[CHUNKS_PER_THREAD];
    #pragma unroll
    for (int idx = 0; idx < CHUNKS_PER_THREAD; ++idx) {
        const int i     = tid + idx * 128;
        const int row   = i / CHUNKS;
        const int chunk = i % CHUNKS;
        offA[idx] = (M0 + row) * K + chunk * 8;
        offB[idx] = (N0 + row) * K + chunk * 8;
    }

    const uint32_t sA0 = (uint32_t)__cvta_generic_to_shared(sA_stages);
    const uint32_t sB0 = (uint32_t)__cvta_generic_to_shared(sB_stages);
    constexpr uint32_t STAGE_BYTES = BM * BK * 2;

    // 发出一个 K-tile 的 global->shared 拷贝到给定 smem 基址 (无 swizzle, 与 base 同布局)
    auto issue_load = [&](uint32_t dA, uint32_t dB, int k_tile) {
        const int k0 = k_tile * BK;
        #pragma unroll
        for (int idx = 0; idx < CHUNKS_PER_THREAD; ++idx) {
            const int i     = tid + idx * 128;
            const int row   = i / CHUNKS;
            const int chunk = i % CHUNKS;
            cp_async_16(dA + (uint32_t)(row * BK + chunk * 8) * 2, &gA[offA[idx] + k0]);
            cp_async_16(dB + (uint32_t)(row * BK + chunk * 8) * 2, &gBT[offB[idx] + k0]);
        }
        cp_async_commit_group();
    };

    // 3 个 stage 的 smem 基址边界 (STAGE_BYTES 为 2 的幂, 前进只需 IADD + 回卷比较)
    const uint32_t sA_end = sA0 + NUM_STAGES * STAGE_BYTES;
    const uint32_t sB_end = sB0 + NUM_STAGES * STAGE_BYTES;

    // Prologue: 提前 issue 前 NUM_STAGES-1 个 K-tile, 流水填满 (s 为展开常量, 无运行时乘)
    #pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; ++s)
        issue_load(sA0 + s * STAGE_BYTES, sB0 + s * STAGE_BYTES, s);
    cp_async_wait_near();   // 等最早的组 (tile 0) 到, 保留最新一组在途
    __syncthreads();

    // 滚动基址: baseX = 当前计算 stage, loadX = 下一待装载 stage.
    // 用 IADD + 回卷比较替代 %NUM_STAGES 与 *STAGE_BYTES, 省掉每 tile 的取模/乘法整数指令.
    uint32_t baseA = sA0;
    uint32_t baseB = sB0;
    uint32_t loadA = sA0 + (NUM_STAGES - 1) * STAGE_BYTES;
    uint32_t loadB = sB0 + (NUM_STAGES - 1) * STAGE_BYTES;

    for (int k_tile = 0; k_tile < NUM_K; ++k_tile) {
        // issue 下一批: tile k_tile + (NUM_STAGES-1) 进刚被释放的 stage
        const int nk = k_tile + (NUM_STAGES - 1);
        if (nk < NUM_K) {
            issue_load(loadA, loadB, nk);
            loadA += STAGE_BYTES; if (loadA == sA_end) loadA = sA0;
            loadB += STAGE_BYTES; if (loadB == sB_end) loadB = sB0;
        }
        cp_async_wait_near();   // 等本 tile 的组完成 (它是当前最旧的)
        __syncthreads();

        // ---------- 计算: 与 base 逐字节相同, smem 基址直接用滚动变量 baseA/baseB ----------
        #pragma unroll
        for (int kk = 0; kk < NUM_KK; ++kk) {
            const int kslice = kk * 16;
            const int grp    = lane >> 3;
            uint32_t aFrag[4][4];
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
        __syncthreads();   // 全部读完本 stage 才允许下轮覆盖它
        baseA += STAGE_BYTES; if (baseA == sA_end) baseA = sA0;
        baseB += STAGE_BYTES; if (baseB == sB_end) baseB = sB0;
    }

    // ---------- epilogue: 与 base 相同, 朴素直接写回 (不合并) ----------
    const int q = lane % 4;
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int row = M0 + mwarp * 64 + mt * 16 + lane / 4;
            const int col = N0 + nwarp * 64 + nt * 8 + 2 * q;
            half* p = &gC[row * N + col];
            // 与 base 相同: 每个 reg (2 个 fp16) 打包成 1 次 32b store
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

    const size_t smem_size = NUM_STAGES * 2 * BM * BK * sizeof(half);
    cudaFuncSetAttribute(gemm_amper, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

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
    printf("gemm_base_multistage: %.3f ms/iter  (%.1f TFLOP/s)\n", ms / 100, tflops);

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
