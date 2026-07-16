# 合法模型旋转与 DCT 潜力复核（2026-07-17）

## 证据边界

- **已有工作**：2026-07-16 的 Llama-2-7B-chat 块级研究，证明原始 channel 顺序上的 DCT 低频裁剪不成立，并观察到 dense Hadamard/FFT 的 q3 重建优势；它不是端到端模型结果。
- **本次新增尝试**：在 Llama-2-7B base 上，将 Identity、1D DCT、Hadamard 和实数 RDFT 放到相同、可逆的在线线性层位置；校准来自 WikiText-2 train，PPL 来自不重叠的 WikiText-2 test 前缀。
- 权重是 fake-quant 后反量化到 FP16，通过普通浮点 GEMM 执行；本文不声称 packed INT3/INT4 存储或整数 kernel 加速。
- 当前模型有 32 个 query heads 和 32 个 KV heads，不使用 GQA；实现按 KV-head 边界编写，但该 checkpoint 不能实证 GQA 收益。

## 核心结论

- 统一 all-input 位置不支持 dense rotation：q3 Identity/DCT 为 9.4400/40.2434，q4 为 6.5672/6.6778。
- DCT 的保留信号局限在 q_proj 单侧输入旋转：Identity/DCT/Hadamard/RDFT PPL 为 6.2544/6.2290/6.2400/6.2412；输出侧 DCT 为 6.2475。
- Permutation 没有稳定超过普通 DCT：输入侧 spectral-DCT 为 6.2391，双侧 spectral-DCT 为 6.2595，后者还需 1466699 metadata bits。
- Head/RoPE-aware DCT 未获支持：attention-head q4 Identity/DCT 为 6.3047/6.3589，RoPE-pair q4 为 6.2194/6.2338。
- 参考实现中 DCT 比 Hadamard 快，但不是零成本：0.2714 ms 对 0.5356 ms；包含 FP16 GEMM 的 q_proj 路径中，Identity/DCT/Hadamard 为 0.0840/0.3574/0.6134 ms。
- 跨测试区段复核中，q_proj 输入 DCT 相对同 scope Identity 的平均 ΔPPL 为 -0.0237，3/3 个区段获胜。

## 统一 all-input 比较

同协议 FP16 基线 PPL：6.195575。

| bits | transform | PPL | ΔPPL | weight rel-MSE | calibration output rel-MSE | tokens/s |
|---|---|---|---|---|---|---|
| — | identity | 6.1956 | 0.0000 | — | — | 3208.8124 |
| 3 | dct | 40.2434 | 34.0478 | 0.0749 | 0.0476 | 3878.3861 |
| 3 | hadamard | 55.3110 | 49.1154 | 0.0748 | 0.0476 | 3945.8640 |
| 3 | identity | 9.4400 | 3.2444 | 0.0774 | 0.0470 | 7166.0920 |
| 3 | rdft | 34.6000 | 28.4044 | 0.0749 | 0.0476 | 5405.4841 |
| 4 | dct | 6.6778 | 0.4822 | 0.0137 | 0.0087 | 4545.6743 |
| 4 | hadamard | 6.7412 | 0.5456 | 0.0137 | 0.0087 | 4477.2774 |
| 4 | identity | 6.5672 | 0.3717 | 0.0143 | 0.0088 | 6983.3481 |
| 4 | rdft | 6.7574 | 0.5618 | 0.0137 | 0.0087 | 5251.9480 |

表中 tokens/s 来自各自独立的完整模型进程，只用于保留运行记录；变换之间的延迟比较以本文后面的 CUDA-event 专项基准为准。

## 单侧/双侧 1D 与 permutation

| scope | bits | transform | permutation | PPL | calibration rel-MSE | metadata bits |
|---|---|---|---|---|---|---|
| q_proj_input | 3 | dct | none | 6.2290 | 0.0176 | 0.0000 |
| q_proj_input | 3 | dct | spectral | 6.2391 | 0.0176 | 733349.6034 |
| q_proj_input | 3 | hadamard | none | 6.2400 | 0.0176 | 0.0000 |
| q_proj_input | 3 | identity | none | 6.2544 | 0.0185 | 0.0000 |
| q_proj_input | 3 | rdft | none | 6.2412 | 0.0176 | 0.0000 |
| q_proj_output_head | 3 | dct | none | 6.2475 | 0.0184 | 0.0000 |
| q_proj_output_head | 3 | hadamard | none | 6.2586 | 0.0184 | 0.0000 |
| q_proj_output_head | 3 | identity | none | 6.2544 | 0.0185 | 0.0000 |
| q_proj_output_head | 3 | rdft | none | 6.2443 | 0.0184 | 0.0000 |
| q_proj_two_sided | 3 | dct | none | 6.2740 | 0.0177 | 0.0000 |
| q_proj_two_sided | 3 | dct | spectral | 6.2595 | 0.0177 | 1466699.2068 |
| q_proj_two_sided | 3 | identity | none | 6.2544 | 0.0185 | 0.0000 |
| q_proj_input | 4 | dct | none | 6.2035 | 0.0032 | 0.0000 |
| q_proj_input | 4 | dct | spectral | 6.2000 | 0.0032 | 733349.6034 |
| q_proj_input | 4 | identity | none | 6.2056 | 0.0035 | 0.0000 |
| q_proj_output_head | 4 | dct | none | 6.2012 | 0.0034 | 0.0000 |
| q_proj_output_head | 4 | identity | none | 6.2056 | 0.0035 | 0.0000 |
| q_proj_two_sided | 4 | dct | none | 6.2153 | 0.0033 | 0.0000 |
| q_proj_two_sided | 4 | dct | spectral | 6.2032 | 0.0032 | 1466699.2068 |
| q_proj_two_sided | 4 | identity | none | 6.2056 | 0.0035 | 0.0000 |

## Head / RoPE 边界

| scope | transform | bits | PPL | calibration rel-MSE |
|---|---|---|---|---|
| attention_head | dct | 4 | 6.3589 | 0.0079 |
| attention_head | hadamard | 4 | 6.3665 | 0.0079 |
| attention_head | identity | 4 | 6.3047 | 0.0079 |
| attention_head | rdft | 4 | 6.3259 | 0.0079 |
| qk_rope_pair | dct | 4 | 6.2338 | 0.0030 |
| qk_rope_pair | identity | 4 | 6.2194 | 0.0030 |

q3 的 2,048-token 筛选如下；它只用于淘汰明显负方案：

| scope | transform | PPL |
|---|---|---|
| attention_head | dct | 20.5040 |
| attention_head | hadamard | 124.4632 |
| attention_head | identity | 4.3677 |
| attention_head | rdft | 31.5830 |
| qk_rope_pair | dct | 4.0388 |
| qk_rope_pair | identity | 3.9892 |

## 跨测试区段稳健性

| scope | transform | permutation | segments | PPL mean | PPL std | mean Δ vs Identity | wins |
|---|---|---|---|---|---|---|---|
| q_proj_input | dct | none | 3 | 5.2779 | 0.8839 | -0.0237 | 3 |
| q_proj_input | dct | spectral | 3 | 5.2888 | 0.8808 | -0.0128 | 3 |
| q_proj_input | hadamard | none | 3 | 5.2772 | 0.8892 | -0.0244 | 3 |
| q_proj_input | identity | none | 3 | 5.3016 | 0.8818 | 0.0000 | 0 |
| q_proj_input | rdft | none | 3 | 5.2896 | 0.8793 | -0.0120 | 3 |
| q_proj_output_head | dct | none | 3 | 5.2942 | 0.8919 | -0.0074 | 2 |
| q_proj_output_head | identity | none | 3 | 5.3016 | 0.8818 | 0.0000 | 0 |

## 参考变换延迟

在线 transform（`[1,256,4096]`，group 128）：

| transform | permutation | median ms | min ms | max ms |
|---|---|---|---|---|
| identity | none | 0.0195 | 0.0184 | 0.0307 |
| dct | none | 0.2714 | 0.2673 | 0.3256 |
| hadamard | none | 0.5356 | 0.5315 | 0.6021 |
| rdft | none | 0.2468 | 0.2437 | 0.3031 |
| dct | per_group_gather | 0.3072 | 0.3041 | 0.3645 |

代表性 4096×4096 q_proj（在线输入 transform + FP16 GEMM）：

| transform | median ms | min ms | max ms |
|---|---|---|---|
| identity | 0.0840 | 0.0829 | 0.0942 |
| dct | 0.3574 | 0.3533 | 4.3786 |
| hadamard | 0.6134 | 0.5929 | 4.6356 |
| rdft | 0.3185 | 0.3154 | 0.3523 |

这些是未融合 PyTorch 参考实现的 CUDA-event 延迟；DCT/RDFT 内部使用 fp32 FFT，Hadamard 使用逐级张量操作。它们用于暴露在线成本，不代表优化后 kernel 排名。

## 判定原则

1. DCT 不再按“低频可裁剪”评价，而按 dense rotation 的量化质量、PPL 与在线成本评价。
2. 单层或局部 MSE 改善若不能转化为同协议 PPL 改善，不作为继续投入依据。
3. permutation 只有在收益超过元数据与 gather 成本时才保留。
4. Head/RoPE-aware 方案必须用同 scope Identity 基线比较；未量化等价性只证明位置合法，不证明量化后有效。
