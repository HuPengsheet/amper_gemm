// 在 gemm_base.cu 基础上加 profile.h 插桩,
// 统计 cp.async / ld.matrix / mma / epilogue 各段的 globaltimer cycles.
// 只在 CTA 0 的 thread 0 上记录, 避免共享输出 buffer.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <climits>
#include <vector>
#include <cuda_fp16.h>

// ProfilerTag 在 profile.h 中只有使用、没有定义, 这里先定义, 再 include profile.h.
enum ProfilerTag : int64_t {
    TAG_CP_ASYNC  = 1,
    TAG_LD_MATRIX = 2,
    TAG_MMA       = 3,
    TAG_EPILOGUE  = 4,
    TAG_MAINLOOP  = 5,
};

#include "profile.h"

#define M 4096
#define N 4096
#define K 4096
#define BM 128
#define BN 128
#define BK 16

// 一次 K-tile: 1 cp.async + 2 ldmatrix + 2 mma = 5 段; 外加 1 epilogue.
// 总 entry ≈ 256*5 + 1 = 1281. mainloop 因 Profiler 的 start/stop 不支持嵌套,
// 用单独的 globaltimer 差值存, 放在每个 slot 的尾部.
#define PROF_NUM_ENTRIES 1500
// 同时记录多少个 CTA (取 thread 0). 每个 CTA 占独立 slot, JSON 里独立 pid/track.
#define N_PROF_CTA 8
// 每 slot 布局 (与 Profiler::init 内部 stride 保持一致):
//   [0]=cnt,
//   [1 .. 1+PROF_NUM_ENTRIES*4-1]=entries,
//   tail 4 个 int64: mainloop 的 sum / min / max / cnt
//     (mainloop = 内层 kk 循环, 用 raw globaltimer 测, 不占 Profiler 槽位).
// init 用的 num_entries 必须把 tail 也算成完整 4-int64 entry, 才能让 stride 对齐.
#define PROF_INIT_NUM_ENTRIES  (PROF_NUM_ENTRIES + 1)
#define SLOT_INT64  (1 + (PROF_INIT_NUM_ENTRIES * 4))
#define ML_SUM_OFFSET  (SLOT_INT64 - 4)
#define ML_MIN_OFFSET  (SLOT_INT64 - 3)
#define ML_MAX_OFFSET  (SLOT_INT64 - 2)
#define ML_CNT_OFFSET  (SLOT_INT64 - 1)

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

__global__ void gemm_amper(const half* __restrict__ gA,
                           const half* __restrict__ gBT,
                           half* __restrict__ gC,
                           int64_t* __restrict__ prof_data)
{
    __shared__ alignas(128) half sA[BM][BK];
    __shared__ alignas(128) half sB[BN][BK];
    const int tid   = threadIdx.x;
    const int lane  = tid % 32;
    const int mwarp = (tid / 32) / 2;
    const int nwarp = (tid / 32) % 2;
    const int M0 = blockIdx.y * BM;
    const int N0 = blockIdx.x * BN;

    // 前 N_PROF_CTA 个 CTA 的 thread 0 持有 Profiler; 其它线程/CTA 走空操作.
    const int bid = blockIdx.y * gridDim.x + blockIdx.x;
    Profiler prof{};
    const bool do_prof = (bid < N_PROF_CTA && tid == 0);
    if (do_prof) prof.init(PROF_INIT_NUM_ENTRIES, prof_data, bid);

    uint32_t C[4][8][2] = {};

    // mainloop (内层 kk 循环) 时长统计: 用 raw globaltimer 测, 不抢 Profiler 槽位.
    int64_t ml_sum = 0, ml_min = INT64_MAX, ml_max = 0, ml_cnt = 0;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---------- cp.async (issue + wait + sync) ----------
        if (do_prof) prof.start(TAG_CP_ASYNC);
        for (int i = tid; i < BM * BK / 8; i += 128) {
            int row  = i / 2;
            int half = i % 2;
            cp_async_16(&sA[row][half * 8], &gA[(M0 + row) * K + k0 + half * 8]);
            cp_async_16(&sB[row][half * 8], &gBT[(N0 + row) * K + k0 + half * 8]);
        }
        cp_async_commit_group();
        cp_async_wait_all();
        __syncthreads();
        if (do_prof) prof.stop();

        // ---------- mainloop (内层 kk 循环: ldmatrix + mma) ----------
        int64_t ml_iter_start = 0;
        if (do_prof) ml_iter_start = globaltimer();

        for (int kk = 0; kk < 2; ++kk) {
            const int kslice = kk * 8;

            // ---------- ld.matrix ----------
            if (do_prof) prof.start(TAG_LD_MATRIX);
            uint32_t aFrag[4][2];
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const uint32_t base = (uint32_t)__cvta_generic_to_shared(&sA[0][0]);
                const int fr = mwarp * 64 + mt * 16;
                const uint32_t addr = base + (uint32_t)(fr + (lane & 15)) * (BK * 2)
                                    + (uint32_t)kslice * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                             : "=r"(aFrag[mt][0]), "=r"(aFrag[mt][1]) : "r"(addr));
            }
            uint32_t bFrag[8];
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const uint32_t base = (uint32_t)__cvta_generic_to_shared(&sB[0][0]);
                const int fr = nwarp * 64 + nt * 8;
                const uint32_t addr = base + (uint32_t)(fr + (lane & 7)) * (BK * 2)
                                    + (uint32_t)kslice * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                             : "=r"(bFrag[nt]) : "r"(addr));
            }
            if (do_prof) prof.stop();

            // ---------- mma ----------
            if (do_prof) prof.start(TAG_MMA);
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt)
                #pragma unroll
                for (int nt = 0; nt < 8; ++nt)
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                        "{%0,%1}, {%2,%3}, {%4}, {%5,%6};"
                        : "=r"(C[mt][nt][0]), "=r"(C[mt][nt][1])
                        : "r"(aFrag[mt][0]), "r"(aFrag[mt][1]), "r"(bFrag[nt]),
                          "r"(C[mt][nt][0]), "r"(C[mt][nt][1]));
            if (do_prof) prof.stop();
        }
        if (do_prof) {
            int64_t d = globaltimer() - ml_iter_start;
            ml_sum += d; ml_cnt++;
            if (d < ml_min) ml_min = d;
            if (d > ml_max) ml_max = d;
        }
        __syncthreads();
    }
    if (do_prof) {
        prof_data[bid * SLOT_INT64 + ML_SUM_OFFSET] = ml_sum;
        prof_data[bid * SLOT_INT64 + ML_MIN_OFFSET] = ml_min;
        prof_data[bid * SLOT_INT64 + ML_MAX_OFFSET] = ml_max;
        prof_data[bid * SLOT_INT64 + ML_CNT_OFFSET] = ml_cnt;
    }

    // ---------- epilogue ----------
    if (do_prof) prof.start(TAG_EPILOGUE);
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
    if (do_prof) { prof.stop(); prof.flush(); }
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

    // Profiler 输出 buffer: 每 CTA 一个 slot, slot 大小见 SLOT_INT64.
    const size_t prof_int64 = (size_t)SLOT_INT64 * N_PROF_CTA;
    int64_t* dProf = nullptr;
    cudaMalloc(&dProf, prof_int64 * sizeof(int64_t));
    cudaMemset(dProf, 0, prof_int64 * sizeof(int64_t));

    dim3 grid(N / BN, M / BM);
    gemm_amper<<<grid, 128>>>(dA, dBT, dC, dProf);
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

    // ---------- 打印 profile 结果 (跨 N_PROF_CTA 聚合) ----------
    std::vector<int64_t> hProf(prof_int64, 0);
    cudaMemcpy(hProf.data(), dProf, prof_int64 * sizeof(int64_t), cudaMemcpyDeviceToHost);

    const char* name[] = { "?", "cp.async", "ld.matrix", "mma", "epilogue" };
    const char* color[] = { "", "good", "terrible", "yellow", "olive" };
    const char* ml_color = "cold";

    const int NTAG = 5;   // 1..4
    int64_t tot[NTAG] = {0};
    int64_t mn[NTAG]  = {INT64_MAX,INT64_MAX,INT64_MAX,INT64_MAX,INT64_MAX};
    int64_t mx[NTAG]  = {0,0,0,0,0};
    int64_t cntT[NTAG] = {0};

    // mainloop (内层 kk 循环) 的 per-CTA 统计存在 slot tail 4 个 int64.
    int64_t ml_sum_cta[N_PROF_CTA] = {0};
    int64_t ml_min_cta[N_PROF_CTA] = {0};
    int64_t ml_max_cta[N_PROF_CTA] = {0};
    int64_t ml_cnt_cta[N_PROF_CTA] = {0};
    int64_t ml_overall_min = INT64_MAX, ml_overall_max = 0, ml_overall_sum = 0;
    int64_t ml_overall_cnt = 0;

    int64_t total_entries = 0;
    for (int b = 0; b < N_PROF_CTA; ++b) {
        const int64_t* slot = &hProf[b * SLOT_INT64];
        const int64_t cnt = slot[0];
        if (cnt <= 0) continue;
        total_entries += cnt;
        for (int64_t i = 0; i < cnt; ++i) {
            const int64_t* e = &slot[1 + i * 4];
            int tag = (int)e[1];
            int64_t dur = e[3];
            if (tag < 1 || tag >= NTAG) continue;
            tot[tag] += dur; cntT[tag]++;
            if (dur < mn[tag]) mn[tag] = dur;
            if (dur > mx[tag]) mx[tag] = dur;
        }
        ml_sum_cta[b] = slot[ML_SUM_OFFSET];
        ml_min_cta[b] = slot[ML_MIN_OFFSET];
        ml_max_cta[b] = slot[ML_MAX_OFFSET];
        ml_cnt_cta[b] = slot[ML_CNT_OFFSET];
        ml_overall_sum += ml_sum_cta[b];
        ml_overall_cnt += ml_cnt_cta[b];
        if (ml_cnt_cta[b] > 0) {
            if (ml_min_cta[b] < ml_overall_min) ml_overall_min = ml_min_cta[b];
            if (ml_max_cta[b] > ml_overall_max) ml_overall_max = ml_max_cta[b];
        }
    }

    if (total_entries == 0) {
        printf("profiler: no entries\n");
    } else {
        printf("\n=== profile (%d CTAs, thread 0 each, globaltimer cycles) ===\n", N_PROF_CTA);
        printf("total entries = %lld\n", (long long)total_entries);
        printf("%-12s %12s %12s %12s %12s %14s\n",
               "tag", "count", "sum", "min", "max", "avg");
        int64_t sum_sub = 0;
        for (int t = 1; t < NTAG; ++t) {
            double avg = cntT[t] ? (double)tot[t] / cntT[t] : 0.0;
            printf("%-12s %12lld %12lld %12lld %12lld %14.1f\n",
                   name[t],
                   (long long)cntT[t],
                   (long long)tot[t],
                   cntT[t] ? (long long)mn[t] : 0,
                   (long long)mx[t],
                   avg);
            sum_sub += tot[t];
        }
        printf("----------------------------------------------------\n");
        // mainloop (per iter)
        double ml_avg_iter = ml_overall_cnt ? (double)ml_overall_sum / ml_overall_cnt : 0.0;
        printf("mainloop per-iter (cycles): avg=%.1f  min=%lld  max=%lld  cnt=%lld\n",
               ml_avg_iter, (long long)ml_overall_min, (long long)ml_overall_max,
               (long long)ml_overall_cnt);
        // mainloop 总 (每 CTA 平均): 单 CTA 的 mainloop 总和 / iter 数 * 一轮平均 cycles, 简化为 sum/CTA
        double ml_avg_per_cta = N_PROF_CTA ? (double)ml_overall_sum / N_PROF_CTA : 0.0;
        printf("mainloop total per-CTA (avg over %d CTAs) = %.1f cycles\n",
               N_PROF_CTA, ml_avg_per_cta);

        // mainloop per-iter 应当 ≈ 2*(LD + MMA) (kk=0, kk=1 各一套).
        double avg_ld  = cntT[TAG_LD_MATRIX] ? (double)tot[TAG_LD_MATRIX] / cntT[TAG_LD_MATRIX] : 0.0;
        double avg_mma = cntT[TAG_MMA]       ? (double)tot[TAG_MMA]       / cntT[TAG_MMA]       : 0.0;
        double ld_mma_per_iter = 2.0 * (avg_ld + avg_mma);
        printf("(LD+MMA) per-iter = 2*(%.1f + %.1f) = %.1f cycles\n",
               avg_ld, avg_mma, ld_mma_per_iter);
        printf("mainloop - (LD+MMA) per-iter = %.1f cycles  (loop/sync 间隙)\n",
               ml_avg_iter - ld_mma_per_iter);

        printf("\nmainloop per-CTA (per-iter stats, cycles):\n");
        printf("%-6s %10s %10s %12s %10s %12s\n",
               "CTA", "min", "max", "sum", "cnt", "avg");
        for (int b = 0; b < N_PROF_CTA; ++b) {
            double avg = ml_cnt_cta[b] ? (double)ml_sum_cta[b] / ml_cnt_cta[b] : 0.0;
            printf("CTA%-4d %10lld %10lld %12lld %10lld %12.1f\n",
                   b,
                   ml_cnt_cta[b] ? (long long)ml_min_cta[b] : 0,
                   ml_cnt_cta[b] ? (long long)ml_max_cta[b] : 0,
                   (long long)ml_sum_cta[b],
                   (long long)ml_cnt_cta[b],
                   avg);
        }

        printf("\nsub-section share (of sub-section sum):\n");
        for (int t = 1; t < NTAG; ++t) {
            printf("  %-10s %10.2f%%\n", name[t],
                   sum_sub ? 100.0 * (double)tot[t] / (double)sum_sub : 0.0);
        }
    }

    // ---------- 写 Perfetto (Chrome Trace Event) JSON ----------
    // profile.h: start=globaltimer()(纳秒), dur=纳秒差; chrome trace 用 us.
    // 多 CTA: 每个 CTA 一个独立 pid (Perfetto 中显示为独立 track).
    {
        FILE* fp = fopen("gemm_profile.json", "w");
        if (!fp) {
            fprintf(stderr, "cannot open gemm_profile.json for write\n");
        } else {
            // 全局时间零点: CTA 0 第一条 entry 的 start.
            int64_t base_ns = 0;
            for (int b = 0; b < N_PROF_CTA; ++b) {
                const int64_t* slot = &hProf[b * SLOT_INT64];
                if (slot[0] > 0) { base_ns = slot[1 + 0 * 4 + 2]; break; }
            }
            fprintf(fp, "{\"traceEvents\":[\n");
            // 每个 CTA 一组 metadata + mainloop envelope (每 outer iter 一条, 从 LD/MMA 合成) + 子段.
            for (int b = 0; b < N_PROF_CTA; ++b) {
                const int64_t* slot = &hProf[b * SLOT_INT64];
                const int64_t cnt = slot[0];
                if (cnt <= 0) continue;
                fprintf(fp,
                    "  {\"name\":\"process_name\",\"ph\":\"M\",\"pid\":%d,"
                    "\"args\":{\"name\":\"CTA%d\"}},\n", b, b);
                fprintf(fp,
                    "  {\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":%d,\"tid\":0,"
                    "\"args\":{\"name\":\"thread0\"}},\n", b);

                // 扫描 entries: 每个 outer iter 形如 cp.async, LD, MMA, LD, MMA.
                // mainloop envelope = 从第一个 LD 的 start 到第二个 MMA 的 start+dur.
                for (int64_t i = 0; i < cnt; /* manual */) {
                    const int64_t* e0 = &slot[1 + i * 4];
                    int tag0 = (int)e0[1];
                    if (tag0 != TAG_LD_MATRIX) { i++; continue; }
                    // e0 = LD(kk=0); 期望后续 MMA, LD, MMA
                    if (i + 3 >= cnt) break;
                    const int64_t* eMMA0 = &slot[1 + (i+1) * 4];
                    const int64_t* eLD1  = &slot[1 + (i+2) * 4];
                    const int64_t* eMMA1 = &slot[1 + (i+3) * 4];
                    if ((int)eMMA0[1] != TAG_MMA ||
                        (int)eLD1[1]  != TAG_LD_MATRIX ||
                        (int)eMMA1[1] != TAG_MMA) { i++; continue; }
                    int64_t ml_start_ns = e0[2];
                    int64_t ml_end_ns   = eMMA1[2] + eMMA1[3];
                    double ts_us  = (double)(ml_start_ns - base_ns) / 1000.0;
                    double dur_us = (double)(ml_end_ns - ml_start_ns) / 1000.0;
                    fprintf(fp,
                        "  {\"name\":\"mainloop\",\"cat\":\"gemm\",\"ph\":\"X\","
                        "\"ts\":%.3f,\"dur\":%.3f,\"pid\":%d,\"tid\":0,"
                        "\"args\":{\"envelope\":true,\"cname\":\"%s\"}},\n",
                        ts_us, dur_us, b, ml_color);
                    i += 4;
                }

                for (int64_t i = 0; i < cnt; ++i) {
                    const int64_t* e = &slot[1 + i * 4];
                    int tag = (int)e[1];
                    if (tag < 1 || tag >= NTAG) continue;
                    double ts_us  = (double)(e[2] - base_ns) / 1000.0;
                    double dur_us = (double)e[3] / 1000.0;
                    fprintf(fp,
                        "  {\"name\":\"%s\",\"cat\":\"gemm\",\"ph\":\"X\","
                        "\"ts\":%.3f,\"dur\":%.3f,\"pid\":%d,\"tid\":0,"
                        "\"args\":{\"sm_id\":%lld,\"cname\":\"%s\"}},\n",
                        name[tag], ts_us, dur_us, b,
                        (long long)e[0], color[tag]);
                }
            }
            // 末尾多了一个 ",\n", 用 fseek 截掉再补 ']' + metadata.
            long pos = ftell(fp);
            fseek(fp, pos - 2, SEEK_SET);   // 覆盖最后的 ",\n"
            fprintf(fp, "\n],\n\"metadata\":{\"cpu_ns\":1,\"unit\":\"us\"}\n}\n");
            fclose(fp);
            printf("\nprofiler: wrote gemm_profile.json (%lld entries, %d CTAs)\n",
                   (long long)total_entries, N_PROF_CTA);
        }
    }

    free(hA); free(hBT); free(hC);
    cudaFree(dA); cudaFree(dBT); cudaFree(dC); cudaFree(dProf);
    return 0;
}
