# Ampere GEMM 优化总结

## 结论

**最终最快版本：`gemm_bk64_multistage_v12.cu` — 1.876~1.906 ms/iter（72~73 TFLOP/s），超过了 cutedsl（1.924 ms / 71.4 TFLOP/s）。**

| 版本 | 配置 | 每 SM 块数 | 时间 | TFLOP/s |
|---|---|---|---|---|
| v8/v9 | BK=64, 3-stage, 1 块/SM | 4 warps | 2.14~2.18 ms | 63~64 |
| v12 | **BK=32, 3-stage, 2 块/SM** | **8 warps** | **1.876~1.906 ms** | **72~73** |
| v13 | BK=32, 2-stage, 3 块/SM | 12 warps | 1.93~1.97 ms | 70~71 |
| cutedsl (CuTeDSL) | — | — | 1.924 ms | 71.4 |

v13（3 块/SM、12 warps）反而比 v12 略慢：2-stage 流水只允许 1 个 cp.async group 在途，`wait_group 0` 让每个 k-tile 都全等加载完成，mem 延迟隐藏能力不如 3-stage 的 2 组在途，抵消了多 4 个 warp 的收益。

## 问题定位：两个 kernel 都是延迟受限，不是带宽/发射受限

- tensor pipe 利用率只有 ~46~50%（约 180 cycles/ktile 的非 tensor 停顿）。
- RTX 5060（sm_120, GB206, 30 SM）fp16 tensor 峰值约 140 TFLOP/s，实测 71 TFLOP/s ≈ 峰值一半。
- 每个 warp 的 HMMA 序列之间有大量停顿：手写 kernel 里 ptxas 在每条 HMMA 后插 `@!UPT UIADD3 URZ,URZ,URZ,URZ` 死 NOP（v8 SASS 293 条里 122 条），cutedsl 的 SASS 用真实 LDSM/LDGSTS/地址计算填满 HMMA 间隙、HMMA 从不背靠背。

## 走过的弯路（结论：都无效或负收益）

1. **去 volatile**（v9）：把 mma asm 的 `asm volatile` 改 `asm`，期望 ptxas 自由调度。结果 SASS 与性能都无变化 —— 死 NOP 是 sm_120 ptxas 的调度行为，与 volatile 无关。
2. **源码层穿插 LDSM**（v10）：在 mma_kk 里每 2 条 HMMA 插 1 条下一 kk 的 ldmatrix。死 NOP 从 122→95，但新增 IADD3/LOP3/IMAD 开销，整体 2.16~2.20 ms，**变差**。
3. **A/B LDGSTS 拆分**（v11）：按 cutedsl 的 DSL 顺序，把 LDGSTS 拆成 HMMA 前发 A、HMMA 后发 B+commit。2.75~6.7 ms，**大幅变差**（手写形状下 A+B 合并发射更优）。

关键认识：**cutedsl 的调度优势是结构性的（predication 代替分支、fragment 寄存器直接复用、流水整体设计），手写 inline asm 的循环形状局部最优，指令级调度上的差距难以从源码控制**。

## 最终起效的改动：v12 = BK=32 + 2 块/SM

针对"延迟受限"的直接攻击：**提高每个 SM 的并发 warp 数来隐藏 HMMA 延迟**。

- BK 64→32，使每块 smem 从 96KB 降到 48KB（3-stage × 2 × 128×32 × 2B）。
- `__launch_bounds__(128, 2)` + 48KB/块 → 2 块/SM = **8 warps/SM**（v8/v9 只有 4 warps），occupancy 翻倍。
- 实测确认 `cudaOccupancyMaxActiveBlocksPerMultiprocessor = 2`（168 regs × 128 × 2 = 43K ≤ 65K regs，48KB × 2 = 96KB ≤ 100KB smem）。
- 修复 BK=32 的 swizzle 溢出 bug：BK=64 的 mask `(row & 7) << 3` 在 64B 行上溢出（swz 最大 56，字节偏移 112+16 > 64B），改为 `SWZ_MASK = CHUNKS-1 = 3`，即 `(row & 3) << 3`，swz ∈ {0,8,16,24} 不超行界。
- Epilogue 沿用 48KB 窗口作 C 转置缓冲（padded stride 272B × 128 行 = 34.8KB，仍装得下）。

## 对比 cutedsl 的访存验证（之前问题的结论）

- global→shared 的 load request 两边一致（~400K/40W 量级）；cutedsl 的 wavefront 数是手写的 4 倍是 async 数据路径的计数产物，**并非合并访存储问题**，两边访存都无冲突、都合并。

## 文件清单

- `gemm_bk64_multistage_v12.cu` — **最终版（最快）**：BK=32, 3-stage, 2 块/SM
- `gemm_bk64_multistage_v13.cu` — BK=32, 2-stage, 3 块/SM（略慢）
- `gemm_bk64_multistage_v9.cu` — BK=64 旧基线（2.14 ms）
- `gemm_bk64_multistage_v10.cu` / `v11.cu` — 失败实验（穿插 LDSM / 拆分 LDGSTS）
- `cutedsl_gemm.py` — cutedsl 基线（1.924 ms）
- `occ_probe.cu` — occupancy 探针（验证 2 块/SM）

## 编译与运行

```bash
nvcc -arch=sm_120 -O3 -Xptxas -v gemm_bk64_multistage_v12.cu -o gemm_bk64_multistage_v12
./gemm_bk64_multistage_v12
```

## 后续可尝试的方向

1. **把 HMMA 的 32 个累加器 C 的寄存器占用（64 regs）压下来**，尝试塞进更多块/SM 或减少 register 压力。
2. **swizzle 之外再调 NUM_STAGES/BK**：BK=16 + NUM_KK=1 需要重构静态双缓冲（当前结构只支持 NUM_KK≥2），若做成单 fragment 缓冲可降到 24KB/块 → 4 块/SM（16 warps），但寄存器要压到 ≤128。
3. 既然 3-stage/2 块已达 cutedsl 水平，进一步提效主要靠消除剩余 ~50% 的 tensor 空闲 —— 即更激进地填满 HMMA 间隙（在 sm_120 上需控制 ptxas 的调度）。
