# 减少 gemm_base_multistage 中整数计算指令的说明

## 1. 背景与结论

在 `gemm_base.cu` 与 `gemm_base_multistage.cu` 的 NCU 对比中,后者出现了两个异常:

1. global store 是前者的两倍 —— 已定位为 epilogue 写回被写错(见第 2 节背景,非本文重点);
2. 整数指令明显偏多,`stall Math pipe throttle`(Math pipe 节流停顿)显著增加。

本文只讲第 2 点:整数指令为什么多、以及如何用**滚动基址指针**把它压下去。

先给结论:多出来的整数指令,根因不是 compute(ldmatrix + mma 两版逐字节相同),而是
**multistage 引入的「运行时 stage 索引」把原本可被编译器折叠的地址计算,退化成了
每 tile 都要现场算的取模 / 乘法 / 64 位指针加法**。

---

## 2. 整数指令的来源分析

对照 `gemm_base` 与 `gemm_base_multistage` 的 SASS(cuobjdump 统计),三个独立来源:

### 2.1 动态 shared memory + 运行时 stage 偏移(LEA / IADD3 / S2R)

`gemm_base` 用静态 smem,`baseA = __cvta_generic_to_shared(&sA[0][0])` 是**编译期常量**,
后续 `ldmatrix` 的地址 `baseA + row*64 + kslice*2 + kOff*2` 里,`baseA`、`kslice`、`kOff`
全是常量,能被立即数折叠,只剩 `row` 相关的少量运算:

```c
// gemm_base.cu
__shared__ alignas(128) half sA[BM][BK];   // 静态 smem,基址编译期已知
```

`gemm_base_multistage` 换成动态 smem + 运行时 stage 偏移:

```c
// gemm_base_multistage.cu(改前)
extern __shared__ __align__(128) half smem_buf[];
...
const uint32_t baseA = sA0 + cur_stage * STAGE_BYTES;   // cur_stage 运行时 -> 每次重算
```

`cur_stage` 是运行时变量,导致每次 `ldmatrix` / `cp.async` 的 shared 地址都必须现场用
LEA/IADD3 重算。动态 smem 本身还要额外 `S2R` 读基址(S2R 从 2 条增至 6 条)。

### 2.2 `% NUM_STAGES` 取模(IMAD / LOP3 / PRMT)

`NUM_STAGES = 3` 不是 2 的幂,编译 `k_tile % 3` 会展开成「魔数乘法 + 移位 + 回减」的一串
整数指令(IMAD.HI 之类),循环 128 次每次都算。SASS 里那 4 条 `PRMT`(字节置换)基本就来自这里。

### 2.3 `* STAGE_BYTES` 运行时乘法(IMAD / LEA)

`issue_load` 里 `st * STAGE_BYTES`、循环头 `cur_stage * STAGE_BYTES`,虽然 `STAGE_BYTES = 8192`
是 2 的幂(本可退化成左移),但被乘数 `st`/`cur_stage` 是运行时值,仍要一条指令。

### 2.4(未处理)64 位全局地址 LEA

`srcA[idx]` / `srcB[idx]` 被预计算成 64 位指针数组,循环里 `srcA[idx] + k0` 是 64 位指针加法
(LEA + LEA.HI.X 成对)。这是 multistage 相对 base 多出的最大单项(LEA 36 vs 17),本文未动,
留作后续优化项。

---

## 3. 优化方法:滚动基址指针

**核心思想**:`% 3` 和 `* STAGE_BYTES` 的本质,是在「等间距的一小组 stage 基址」之间循环计数。
既然 stage 基址是公差为 `STAGE_BYTES`(2 的幂)的循环序列,就用**累加 + 回卷**代替取模和乘法。

### 3.1 三处改动

**改动一:`issue_load` 直接收 smem 基址,而非 stage 序号**,消除函数内的 `st * STAGE_BYTES`:

```c
// 改前
auto issue_load = [&](int st, int k_tile) {
    const uint32_t dA = sA0 + st * STAGE_BYTES;   // 运行时乘法
    const uint32_t dB = sB0 + st * STAGE_BYTES;
    ...
};

// 改后
auto issue_load = [&](uint32_t dA, uint32_t dB, int k_tile) {
    ... // 直接用 dA / dB
};
```

**改动二:prologue 用 `#pragma unroll` 展开**,`s * STAGE_BYTES` 的 `s` 变成编译期常量,
不产生运行时乘法:

```c
#pragma unroll
for (int s = 0; s < NUM_STAGES - 1; ++s)
    issue_load(sA0 + s * STAGE_BYTES, sB0 + s * STAGE_BYTES, s);
```

**改动三:主循环维护 4 个滚动基址指针**,替代 `% NUM_STAGES` 与 `cur_stage * STAGE_BYTES`:

| 变量 | 含义 |
|------|------|
| `baseA` / `baseB` | 当前要**计算**的 stage 基址 |
| `loadA` / `loadB` | 下一个要**装载**的 stage 基址 |

```c
const uint32_t sA_end = sA0 + NUM_STAGES * STAGE_BYTES;
const uint32_t sB_end = sB0 + NUM_STAGES * STAGE_BYTES;

uint32_t baseA = sA0;
uint32_t baseB = sB0;
uint32_t loadA = sA0 + (NUM_STAGES - 1) * STAGE_BYTES;   // = 待装载的第 3 个 stage
uint32_t loadB = sB0 + (NUM_STAGES - 1) * STAGE_BYTES;

for (int k_tile = 0; k_tile < NUM_K; ++k_tile) {
    const int nk = k_tile + (NUM_STAGES - 1);
    if (nk < NUM_K) {
        issue_load(loadA, loadB, nk);
        loadA += STAGE_BYTES; if (loadA == sA_end) loadA = sA0;   // 前进 + 回卷
        loadB += STAGE_BYTES; if (loadB == sB_end) loadB = sB0;
    }
    cp_async_wait_near();
    __syncthreads();

    // ... 计算段直接使用 baseA / baseB(不再每次现算基址) ...

    __syncthreads();
    baseA += STAGE_BYTES; if (baseA == sA_end) baseA = sA0;       // 前进 + 回卷
    baseB += STAGE_BYTES; if (baseB == sB_end) baseB = sB0;
}
```

### 3.2 为什么这样更省

- `x += STAGE_BYTES`:`STAGE_BYTES = 8192` 是 2 的幂,编译成一条 `IADD3`(加立即数)。
- 回卷 `if (x == end) x = base`:`k_tile` 是 warp-uniform 的值,编译器把它放到
  **uniform datapath**(`UISETP` + `USEL`/`UMOV`),不占用线程 math pipe。
- 对照取模 `% 3`:非 2 的幂,是「魔数乘法 + 移位 + 回减」约 3~4 条,全在普通 datapath。

等价性:前进 3 次恰好 `== sA_end`(`STAGE_BYTES` 对齐、`sA0` 128 字节对齐),回卷精确,
与原来的 `t % 3` 完全一致,`host check` 验证 `mismatch 0`。

---

## 4. 效果(SASS 中普通 datapath 整数指令)

`cuobjdump -sass` 统计对比(只列整数类、非 uniform 指令):

| 指令 | base | multistage 改前 | multistage 改后 | 说明 |
|------|------|-----------------|-----------------|------|
| IMAD  | 86 | 85 | **70** | 取模/乘法的魔数乘法被消除 |
| IADD3 | 20 | 31 | 41 | 滚动前进 `+= STAGE_BYTES` |
| LEA   | 17 | 36 | 36 | 64 位地址,未处理 |
| LOP3  | 22 | 22 | **18** | |
| SHF   | 20 | 20 | **18** | |
| PRMT  | 0  | 4  | **0**  | 取模的字节置换彻底消失 |
| S2R   | 2  | 6  | 6  | 动态 smem 基址读取 |
| 合计(普通 datapath) | 177 | 206 | **191** | **-15** |

同时,`USEL`/`UMOV`/`UISETP`/`UIADD3` 等 uniform 指令从约 4 条增至约 20 条,
这些落在 uniform datapath,**不挤 math pipe**。

**净效果**:

1. `% NUM_STAGES` 取模完全消除(PRMT 归零、IMAD 降 15);
2. 挤 math pipe 的整数指令净减 15 条(206 → 191);
3. 约 16 条指令从 math pipe 转移到 uniform pipe。

这与之前观察到的 `stall Math pipe throttle` 下降方向一致。

---

## 5. 遗留项

- **LEA(64 位全局地址)**仍是 multistage 相对 base 最大的整数开销(36 vs 17),来源是
  `srcA[idx]`/`srcB[idx]` 预计算指针数组。后续可改为「保存 32 位偏移 + 最后一次 64 位加」,
  或直接回到 base 的 32 位 `(M0+row)*K` 偏移写法,预计可把 LEA 降到 20 左右。
- 本次改动**只做减法**:未改 stage 数、未改装载粒度、未动 compute / swizzle / epilogue 逻辑,
  因此 multistage 相对 base 的「occupancy 减半 + ldmatrix 延迟裸露」等结构性问题依旧存在,
  不会因为本次优化而变成比 base 快。