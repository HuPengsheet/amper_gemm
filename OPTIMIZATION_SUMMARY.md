# GEMM 优化演进总结

> 从"朴素可正确"到"接近峰值"，4 个版本逐步加东西，**每一步实测都提速**。
> 用一句话概括演进：**先修掉写回的碎片事务，再一次性把 global 延迟、ldmatrix
> 延迟、bank conflict 三层瓶颈全部藏住，最后工程化收尾。**
>
> 环境: RTX 4060 Laptop (sm_89, Ada), CUDA 12.5, 矩阵 4096×4096×4096, fp16 累加。
> 基准数据为同一机器上多次测量的稳定区间（每版本内部循环 100 次取均值）。

---

## 1. 总览

所有版本 tiling 完全一致（与最终版 gemm_finial 对齐）：

| 参数 | 值 | 说明 |
|---|---|---|
| CTA tile | BM × BN = 128 × 128 | 每个 block 算一个 128×128 输出块 |
| K-tile | BK = 32 | 每轮装载 32 列 K |
| 线程 | 128 / block | 4 warps，2×2 布局 |
| MMA | m16n8k16 fp16 | 每 warp 每 kk 做 4×8 = 32 个 mma |
| ldmatrix | m8n8.x4 | 每次取一个 8×8 象限，x4 取 4 个 |
| 全局装载 | cp.async 16B | global → shared 异步拷贝 |

### 4 个文件的演进关系

```
gemm_base ──────────────► gemm_epilogue ──────► gemm_multistage_register ──► gemm_finial
 朴素基线                  合并写回(小步)          流水+双缓冲+swizzle(大步)    生产版(≈step3)
```

> 另有 **`gemm_base_multistage.cu`（对照实验，不在单调链里）**：base 只加 multistage
> 流水，其余不动。实测**变慢**（见 §4），作为"为什么流水必须和双缓冲打包"的实证。

### 性能演进（ms/iter，越小越好；同一轮内对比稳定）

> 笔记本 GPU 温度/时钟波动，跨轮次绝对数字有 ±10% 漂移，**同一轮内相对顺序稳定**。
> 以下为一轮内连续测量。

```
gemm_base               3.61~3.62 ms  38 TFLOP/s   ← 起点：单缓冲 + 无 swizzle + 朴素写回
gemm_epilogue           3.37~3.48 ms  40 TFLOP/s   ← 合并写回 (+用 -maxrregcount 128 钉住占用率)
gemm_multistage_register 2.44~2.58 ms  55 TFLOP/s  ← 流水 + 寄存器双缓冲 + swizzle 合一 (~1.4x)
gemm_finial             2.42~2.55 ms  55 TFLOP/s   ← = step3 + 工程调优 (等价)
```

> ⚠️ 最重要的一条教训：**"每步解决一个问题"不等于"每步都提速"**。
> 优化不是单调的——一个 fix 只有当它恰好命中当前瓶颈时才提速；否则只是把延迟换个
> 地方藏，甚至倒贴资源。本套 4 个文件是刻意**筛选过的单调序列**：
> 只把"能在当前状态兑现"的优化排进来，兑现不了的合到后面一起做。
> 这一点在第 3、4 节的逐版本分析里看得最清楚。

### 资源占用（ptxas 实测）

> 寄存器预算 = 65536 / (128 线程 × 块/SM)。除 gemm_finial 自带 `__launch_bounds__`
> 外，其余不加 launch_bounds，块/SM 由资源自然决定（寄存器与 smem 取较小者）。

| 版本 | 寄存器 | smem/块 | 块/SM | 说明 |
|---|---|---|---|---|
| base | 121 | 16KB 静态 | 4 | 寄存器允许 4 块（4.23） |
| epilogue | 123 | 16KB 静态 | 4 | **必须 `-maxrregcount 128`**：自然分配 132 → 3.88 → 3 块，压到 123 → 4 块（0 spill） |
| multistage_register | 168 | 48KB 动态 | 2 | smem 限制（48KB → 2 块） |
| finial | 168 | 48KB 动态 | 2 | 自带 `__launch_bounds__(128,2)` 明确目标 |

> 注：6 个版本全部 0 spill（gemm_multistage_register / gemm_finial 用动态 smem，
> 需 `cudaFuncSetAttribute` 开 MaxDynamicSharedMemorySize）。不加 launch_bounds 就
> 没有寄存器硬上限，编译器按调度自由分配；代价是像 swizzle/epilogue 这种天然多占
> 寄存器的优化会自己让出块数——这正是 epilogue 必须配 `-maxrregcount` 的原因。

---

## 2. 为什么是这 4 步（而不是"每个问题一步"）

GEMM 的三个经典问题按"藏延迟的优先级"排序：

```
优先级 1: global 装载延迟（cp.async 单缓冲时完全暴露，占比最大）
优先级 2: ldmatrix 延迟（只有 global 藏住后才会浮出来）
优先级 3: bank conflict（ldmatrix 读 smem 的冲突，只在 LDSM 成为关键路径时才算数）
```

加上一个横切的约束：**写回碎片**（epilogue 的全局 store 不合并）。

理论上"每个问题一步"最干净，但实测发现两个单步是"修对地方但当时不兑现"：

- **单独 swizzle**：base 里 bank conflict 只有 2-way（ldmatrix 每行多 1 个周期），
  而内核是单缓冲、全局延迟完全暴露，LDSM 的时间本来就藏在等数据里，消除冲突
  省不到关键路径上的时间；swizzle 的 XOR 寻址反而让地址非仿射、多占 ~11 个寄存器，
  掉 1 块占用率，整核更慢。
- **单独 multistage**：流水把 global 延迟藏住了，但 base 本来就在用 4 块/SM 的
  块间重叠藏同一份延迟，流水的收益是冗余的；反而 smem 16→48KB 把块/SM 压到 2、
  新暴露的 ldmatrix 延迟在低占用率下裸奔，整核更慢。实测 base 3.4ms → base+3-stage
  4.2ms（-22%），见 §4 对照实验。

所以本套 4 步的选法是：**把写回这个独立的小问题单独做成一步（它只碰 epilogue、
不动装载/计算，收益虽小但可兑现）；把三个"互相嵌套的延迟问题"合并成一步，
等三者能同时藏住时一起兑现（这才是真正的 1.4x 大跳跃）。**

---

## 3. 各版本详解

### gemm_base — 朴素基线（什么都不做）

最朴素写法，刻意暴露全部问题：

1. **单缓冲、串行流水**：每个 K-tile 都是
   `cp.async 装载 → wait_all → __syncthreads → 计算 → __syncthreads`。
   global→shared 的拷贝延迟完全暴露，与计算零重叠。
2. **无 swizzle，bank conflict**：shared 直接行优先排布，行宽 64B = 16 word。
   行 r 首字 bank = (r×64)/4 % 32 = (16r)%32，只由 r 奇偶决定。
   ldmatrix 读一个 8×8 矩阵的 8 行时，行 0&4、2&6 撞同一 bank（2-way 冲突）。
3. **写回不合并**：epilogue 按 mma fragment 布局直接写全局，每线程每次只写
   2 个 fp16 (4B)，warp 内 32 线程写的是 16 个不同行上错位的 4B，
   128B 全局事务被严重浪费。

基准 ~3.6 ms / 38 TFLOP/s，4 块/SM。

### gemm_epilogue — 合并写回（第一个真正兑现的小步）

**只改 epilogue**，tiling/装载/计算与 base 逐字节相同：

```
[1] reg → SMEM : 按 mma fragment 布局写进一个 128 列宽的 C 缓冲
[2] SMEM → GMEM: __syncthreads 后每线程按"行内连续 16B"重读，
                 uint4 (128-bit) 合并写回 gC。32 线程覆盖 512B 连续地址，
                 warp 的 store 恰好填满 4 个 128B 事务，完全合并。
```

实现细节：

- 复用 A/B 的 16KB 静态窗口做 C 缓冲，不额外申请 smem，保持 4 块/SM 的对比公平。
  C tile 128×128 = 32KB 放不进 16KB，分两半：pass 0 写 mwarp 0 的 C 行 [0,64)，
  pass 1 写 mwarp 1 的 C 行 [64,128)，两次都写窗口物理行 [0,64)，靠两次
  `__syncthreads` 保证"写入完成 → 读走 → 覆盖"。
- 寻址全部用 uint32 单位（行 stride = 64 u32），写阶段只留 1 个 base 指针，
  读回只留 1 个 smem + 1 组 gmem 指针，寄存器占用最小化。

**为什么单独这一步只能是小步 + 必须配 `-maxrregcount 128`：**

- 写回在整个内核里只占很小的比例，base 的碎片写回浪费的是几个 % 的时间，
  合并后也就省这几个 %——它是真实收益，但很小。
- 更关键的是 smem 中转比 base 的直接 4B 写回多占 ~11 个寄存器（写阶段 +
  读回指针，在计算循环尾部和地址寄存器在边界上叠加）。不加约束 ptxas 自然
  分配 132 regs → 65536/(128×132) = 3.88 → **掉到 3 块/SM**，占用率损失直接
  盖过写回收益，实测 ~4.3ms **比 base 还慢**。
- 用 `-maxrregcount 128`（**编译 flag，不是源码里的 `__launch_bounds__`**）
  把分配压到 123 regs（0 spill），保持 4 块/SM，合并写回才兑现为
  **~3.4ms < base ~3.6ms** 的真实提速。

这是整套序列里最"薄"的一步，但它诚实展示了：**优化要保住占用率才兑现**，
这也是后面 swizzle/epilogue 类"天然多占寄存器"的优化反复出现的主题。

### gemm_multistage_register — 流水 + 寄存器双缓冲 + swizzle（大跳跃）

在 gemm_epilogue 基础上一次性补齐三件事，这是最大的一次跳跃（~1.4x）：

- **[A] cp.async 3-stage 流水**：prologue 提前 issue 前 2 个 K-tile 的拷贝；
  主循环 `wait_group 1` 等最早那组 → 装载 fragment → issue 下一 K-tile 到刚释放
  的 stage → mma 计算。global→shared 拷贝延迟从此与计算重叠。
- **[B] 寄存器双缓冲**：用两套 fragment 寄存器（`aBank[2]/bBank[2]`）交替，
  计算当前 kk 的 mma 时同时把下一个 kk 的 fragment 预装载进另一套；
  **ldmatrix 的延迟被 mma 完全隐藏**（warp 内自掩盖，不依赖占用率）。
  kk 循环显式展开，静态选择 CUR/NXT bank，避免运行时索引。
- **[C] swizzle**：此时 global/ldmatrix 延迟都藏住了，LDSM 的 bank conflict
  才浮出来成为关键路径。用 `(row>>1)&3` 的 XOR swizzle 消除冲突；
  偏移在进主循环前一次算好（`precompute_ldsm_offsets`），热循环里不再重算，
  **不占寄存器，也不掉占用率**。

为什么三点必须一起出现：

- 单独 multistage 会因寄存器暴涨（整块 fragment 数组 → 168 regs）掉到 2 块/SM，
  且 16 次 ldmatrix.x4 背靠背发出、没有 mma 掩盖 → ldmatrix 延迟完全暴露 → 更慢。
- 单独 swizzle 在延迟未藏住时也不兑现（见 base 版分析）。
- 三者合体后：global 装载被流水藏住、ldmatrix 被双缓冲藏住、bank conflict 被
  swizzle 消掉，mma 占满 tensor core，流水才完整兑现。

实测 ~2.5ms / 55 TFLOP/s。这是 4 步里唯一的"大爆发"，前面所有小步攒下的
收益（合并写回、占位逻辑）也一并兑现。

### gemm_finial — 生产版

与 gemm_multistage_register 同一内核，差别是工程化调优：

- `__launch_bounds__(128, 2)` 明确 2 块/SM 目标（与 smem 天然限制一致），
  编译器放心规划寄存器；
- `cudaFuncAttributePreferredSharedMemoryCarveout = 100` 把 smem 切给 L1 更多；
- 源指针按 BK 推进，减少地址计算。

实测 2.42~2.55ms，与 gemm_multistage_register 等价（运行噪声 ±5%）。它是这套
演进的"终点"：完整拥有流水 + 双缓冲 + swizzle + 合并写回全部特性。

---

## 4. 数据与原因对照

| 版本 | ms/iter | 相对 base | 主要瓶颈（修/引入的） |
|---|---|---|---|
| base | 3.61~3.62 | 1.00x | 无流水 + 写回不合并 + bank 冲突（都被延迟藏着） |
| epilogue | 3.37~3.48 | 0.94x | 修掉写回碎片（小收益）；必须 `-maxrregcount 128` 保 4 块/SM |
| multistage_register | 2.44~2.58 | 0.69x | 流水 + 双缓冲 + swizzle 合一，三层延迟同时藏住 → 大跳跃 |
| finial | 2.42~2.55 | 0.68x | 生产版，与 step3 等价 |

### 为什么每个 fix 有的赢有的不赢

- **瓶颈的"优先级"是嵌套的**：global 装载延迟 > ldmatrix 延迟 > bank conflict。
  只有把上层（global）盖住，下层（ldmatrix/bank）才会浮出来成为瓶颈，然后才轮到
  下层优化兑现。base 版单独 swizzle 不赢、单独 multistage 不赢，都是因为盖住的
  不是当时最上面的那层。
- **占用率是隐形瓶颈**：任何优化如果让寄存器/smem 暴涨导致块/SM 下降，都可能把
  收益吃掉（epilogue 132→123 regs 的差别就是 3 块 vs 4 块；multistage 48KB smem
  直接锁死 2 块）。不加 launch_bounds 时块/SM 完全由资源自然决定：
  `⌊65536/(128×regs)⌋` 与 `⌊smem/SMEM_PER_BLOCK⌋` 取小。
- **"单调"是筛选出来的，不是自动的**：真正每次"修一个问题"的朴素序列（base →
  swizzle → epilogue → multistage → register）在第 4 步之前反而一路变慢。本套 4
  步刻意把"当时不兑现"的优化合并到能兑现的时刻，才得到每步都提速的结果。
- **优化要对着数据说话**：无 ncu 时，用"同 tiling 下逐版本改一处"的对照实验就能
  定位瓶颈，这也是这套演化版本存在的意义。

### 对照实验 — base + 单独 multistage（`gemm_base_multistage.cu`）

研究问题：base 只加 multistage 流水（compute/epilogue 逐字节不动、无 swizzle、无
双缓冲），能不能提速？答案：**不能，stage 越多越慢。**

`NUM_STAGES` 从 1（=base）加到 3，只有 smem 变大、占用率变小，流水结构不变：

| NUM_STAGES | smem/块 | 块/SM | warp/SM | ms/iter | 相对 base |
|---|---|---|---|---|---|
| 1（= base） | 16KB | 4 | 16 | 3.38~3.48 | 1.00x |
| 2 | 32KB | 3 | 12 | 3.80~3.85 | ~1.12x |
| 3 | 48KB | 2 | 8 | 4.14~4.22 | ~1.22x |

（改动方法：只改 `#define NUM_STAGES`，同一内核。）

为什么藏住了 global 延迟却更慢：

1. **base 本来就在用占用率藏 global 延迟，流水的收益是冗余的**。base 块内是串行的
   （load→wait→compute→sync），但 SM 上 4 块 = 16 个 warp，一个块等 cp.async 时
   其它块在算。multistage 把这层延迟换到块内藏，但这份延迟 base 靠块间重叠就拿
   到了，等于白干。
2. **流水的代价是占用率减半**：smem 16→48KB，块/SM 4→2、16→8 warp。没有足够的
   块间冗余去藏任何延迟。
3. **新暴露 ldmatrix 延迟在低占用率下裸奔**：每 kk "现场装载 fragment → 立即 mma"，
   mma 依赖 ldmatrix 结果。4 块/SM 时靠别的 warp 盖住，2 块/SM（每调度器仅 2 warp）
   时没人盖。
4. **反证**：`gemm_multistage_register` 同样 48KB smem、2 块/SM，却 4.2ms → 2.4ms。
   差的只是双缓冲的"warp 内自隐藏"——证明 2 块/SM 下唯一能救 ldmatrix 的就是它。

结论：multistage 单独把 base 已靠占用率拿到的收益，换成一种更贵且会暴露新延迟的
实现。**它只有和寄存器双缓冲打包（warp 内自隐藏，不依赖占用率）时，才第一次真正
优于 base 的"堆 warp"策略。**

---

## 5. 如何复现

```bash
NVCC=/usr/local/cuda-12.5/bin/nvcc
$NVCC -arch=sm_89 -O3 gemm_base.cu -o build_gemm_base
# 注意: gemm_epilogue 必须带 -maxrregcount 128，否则 3 块/SM 反而变慢
$NVCC -arch=sm_89 -O3 -maxrregcount 128 gemm_epilogue.cu -o build_gemm_epilogue
$NVCC -arch=sm_89 -O3 gemm_multistage_register.cu -o build_gemm_multistage_register
$NVCC -arch=sm_89 -O3 gemm_finial.cu -o build_gemm_finial
for f in base epilogue multistage_register finial; do ./build_gemm_$f; done
# 输出 ms/iter + host check；同轮内对比验证 base > epilogue > multistage_register ≈ finial

# 对照实验（base + 单独 multistage, 预期比 base 慢; 改 NUM_STAGES 1/2/3 看趋势）
$NVCC -arch=sm_89 -O3 gemm_base_multistage.cu -o build_gemm_base_multistage
./build_gemm_base_multistage
```

每个版本自带 host check：`A[i][k] = 2^(i%4) ∈ {1,2,4,8}`, `B[k][n] = 1`
⇒ `C[i][n] = 4096 × 2^(i%4)`（fp16 精确，逐元素相等校验）。
