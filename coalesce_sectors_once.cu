// coalesce_sectors_once.cu
// 单 warp、每个 kernel 只发一次读命令, 用 ncu 直接看每条读指令的 sector 数。
// (没有循环, 计数器直接就是"每条指令"的值, 不用除以迭代次数。)
//
// 模式1 p1_contig4    32 线程各读 4B(float),  合起来连续 128B
//                     -> 4 sector   (128B / 32B)
// 模式2 p2_contig16   32 线程各读 16B(float4), 合起来连续 512B
//                     -> 16 sector  (512B / 32B)
// 模式3 p3_scattered  32 线程各读 4B, 每线程相隔 512B, 全不连续
//                     -> 32 sector  (每个 4B 独占一个 32B sector, 最差)
// 模式4 p4_grouped32  4 组 x 8 线程, 每组 32B 连续且 32B 对齐, 组间留 256B 缝隙
//                     -> 4 sector / 4 request (每组成独立 sector, 各在不同 128B 行)
// 模式5 p5_grouped16  8 组 x 4 线程, 每组 16B 连续且 32B 对齐, 组间留 256B 缝隙
//                     -> 8 sector / 8 request (每组成独立 sector, 各在不同 128B 行)
// 模式6 p6_broadcast  32 线程都读同一个 float(广播)
//                     -> 1 sector (全部线程命中同一 32B sector)
//
// 结论: sector 按 32B 粒度算, 只要组对齐到 32B 边界, 组越小 -> sector 越多、
//      每个 sector 利用率越低。模式5 是模式4 的"组减半"版本, sector 翻倍。

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define WARP 32

// 模式1: 32 线程各读 4B, 连续 128B
__global__ void p1_contig4(const float* __restrict__ src, float* __restrict__ dst)
{
    const int i = threadIdx.x;
    dst[i] = src[i];   // 一次 4B 读
}

// 模式2: 32 线程各读 16B(float4), 连续 512B
__global__ void p2_contig16(const float4* __restrict__ src, float4* __restrict__ dst)
{
    const int i = threadIdx.x;
    dst[i] = src[i];   // 一次 16B 读
}

// 模式3: 32 线程各读 4B, 线程 i 与 i+1 相隔 512B
__global__ void p3_scattered(const float* __restrict__ src, float* __restrict__ dst)
{
    const int i = threadIdx.x;
    dst[i] = src[(size_t)i * 128];   // 每线程独立 32B sector
}

// 模式4: 4 组 x 8 线程, 每组 32B 连续且 32B 对齐, 组间留 256B 缝隙
__global__ void p4_grouped32(const float* __restrict__ src, float* __restrict__ dst)
{
    const int i = threadIdx.x;
    const int g = i >> 3;   // 组号 0..3
    const int o = i & 7;    // 组内偏移 0..7
    // 组 g 起点字节 = g*(32+256) = 288g, 覆盖 byte [288g, 288g+32) = float [72g, 72g+8)
    dst[i] = src[(size_t)g * 72 + o];
}

// 模式5: 8 组 x 4 线程, 每组 16B 连续且 32B 对齐, 组间留 256B 缝隙
__global__ void p5_grouped16(const float* __restrict__ src, float* __restrict__ dst)
{
    const int i = threadIdx.x;
    const int g = i >> 2;   // 组号 0..7
    const int o = i & 3;    // 组内偏移 0..3
    // 组 g 起点字节 = g*(16+256) = 272g, 覆盖 byte [272g, 272g+16) = float [68g, 68g+4)
    dst[i] = src[(size_t)g * 68 + o];
}

// 模式6: 32 线程都读同一个 float (广播), 只触 1 个 sector
__global__ void p6_broadcast(const float* __restrict__ src, float* __restrict__ dst)
{
    const int i = threadIdx.x;
    dst[i] = src[0];   // 所有线程命中同一地址
}

// ---------- 辅助 ----------

// 源指针索引: 模式1/3/4/5 用; 模式2 是 float4 视图, 单独校验
static size_t src_idx(int mode, int i)
{
    if (mode == 1) return i;
    if (mode == 3) return (size_t)i * 128;
    if (mode == 4) { const int g = i >> 3, o = i & 7; return (size_t)g * 72 + o; }
    { const int g = i >> 2, o = i & 3; return (size_t)g * 68 + o; }   // mode 5
}

// 256B 对齐分配
static void* aligned_malloc(size_t bytes, size_t alignment, void** raw)
{
    if (cudaMalloc(raw, bytes + alignment) != cudaSuccess) { *raw = nullptr; return nullptr; }
    const size_t raw_addr     = (size_t)*raw;
    const size_t aligned_addr = (raw_addr + alignment - 1) & ~(alignment - 1);
    return (void*)aligned_addr;
}

int main()
{
    const size_t n_floats = 4096;  // 模式3 最大 31*128=3968; 模式5 最大 68*7+3=479
    const size_t n_dst    = 256;   // 缓冲富余: p2 写 128, 其余只需 32

    float* h_src = (float*)malloc(n_floats * sizeof(float));
    for (size_t i = 0; i < n_floats; ++i) h_src[i] = (float)(i % 7);

    const size_t alignment = 256;
    void *raw_src, *raw_dst;
    float* d_src = (float*)aligned_malloc(n_floats * sizeof(float), alignment, &raw_src);
    float* d_dst = (float*)aligned_malloc(n_dst    * sizeof(float), alignment, &raw_dst);
    if (!d_src || !d_dst) { fprintf(stderr, "alloc fail\n"); return 1; }
    cudaMemcpy(d_src, h_src, n_floats * sizeof(float), cudaMemcpyHostToDevice);

    printf("== 单 warp 6 种读模式, 每个 kernel 只发一次读指令 ==\n");
    printf("  p1_contig4    : 连续 4B, 128B        -> 4  sector\n");
    printf("  p2_contig16   : 连续 16B, 512B       -> 16 sector\n");
    printf("  p3_scattered  : 分散 4B, 每线程隔512B -> 32 sector\n");
    printf("  p4_grouped32  : 4组x8, 32B对齐, 组隙256B -> 4  sector\n");
    printf("  p5_grouped16  : 8组x4, 16B对齐, 组隙256B -> 8  sector (半满)\n");
    printf("  p6_broadcast  : 32线程读同一个float     -> 1 sector\n\n");
    printf("ncu (需 sudo; 每 kernel 只有一条读指令, 计数器即每指令值):\n");
    printf("  sudo ncu --kernel-name-base function --launch-skip 0 \\\n");
    printf("      --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\\\n");
    printf("                l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\\\n");
    printf("                l1tex__data_pipe_lsu_wavefronts_mem_global_op_ld.sum \\\n");
    printf("      ./build_coalesce_sectors_once\n\n");

    float h_dst[256];
    size_t bad;

    // 模式1
    p1_contig4<<<1, WARP>>>(d_src, d_dst);
    cudaMemcpy(h_dst, d_dst, n_dst * sizeof(float), cudaMemcpyDeviceToHost);
    bad = 0; for (int i = 0; i < WARP; ++i) if (h_dst[i] != h_src[src_idx(1, i)]) ++bad;
    printf("  p1_contig4    : verify %s (bad=%zu)\n", bad ? "FAIL" : "PASS", bad);

    // 模式2
    p2_contig16<<<1, WARP>>>((const float4*)d_src, (float4*)d_dst);
    cudaMemcpy(h_dst, d_dst, n_dst * sizeof(float), cudaMemcpyDeviceToHost);
    bad = 0; for (size_t i = 0; i < WARP * 4; ++i) if (h_dst[i] != h_src[i]) ++bad;
    printf("  p2_contig16   : verify %s (bad=%zu)\n", bad ? "FAIL" : "PASS", bad);

    // 模式3
    p3_scattered<<<1, WARP>>>(d_src, d_dst);
    cudaMemcpy(h_dst, d_dst, n_dst * sizeof(float), cudaMemcpyDeviceToHost);
    bad = 0; for (int i = 0; i < WARP; ++i) if (h_dst[i] != h_src[src_idx(3, i)]) ++bad;
    printf("  p3_scattered  : verify %s (bad=%zu)\n", bad ? "FAIL" : "PASS", bad);

    // 模式4
    p4_grouped32<<<1, WARP>>>(d_src, d_dst);
    cudaMemcpy(h_dst, d_dst, n_dst * sizeof(float), cudaMemcpyDeviceToHost);
    bad = 0; for (int i = 0; i < WARP; ++i) if (h_dst[i] != h_src[src_idx(4, i)]) ++bad;
    printf("  p4_grouped32  : verify %s (bad=%zu)\n", bad ? "FAIL" : "PASS", bad);

    // 模式5
    p5_grouped16<<<1, WARP>>>(d_src, d_dst);
    cudaMemcpy(h_dst, d_dst, n_dst * sizeof(float), cudaMemcpyDeviceToHost);
    bad = 0; for (int i = 0; i < WARP; ++i) if (h_dst[i] != h_src[src_idx(5, i)]) ++bad;
    printf("  p5_grouped16  : verify %s (bad=%zu)\n", bad ? "FAIL" : "PASS", bad);

    // 模式6
    p6_broadcast<<<1, WARP>>>(d_src, d_dst);
    cudaMemcpy(h_dst, d_dst, n_dst * sizeof(float), cudaMemcpyDeviceToHost);
    bad = 0; for (int i = 0; i < WARP; ++i) if (h_dst[i] != h_src[0]) ++bad;
    printf("  p6_broadcast  : verify %s (bad=%zu)\n", bad ? "FAIL" : "PASS", bad);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    free(h_src);
    cudaFree(raw_src);
    cudaFree(raw_dst);
    return 0;
}
