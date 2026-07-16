# 正交变换压缩协议

本协议用于 Hadamard、DCT、FFT、learned rotation 和 Kuramoto-inspired
方案。满足以下条件后，结果才可以进入模型压缩结论。

## 1. 功能等价与计算图

- 明确写出分析变换、逆变换及其作用轴。
- 说明输入旋转和输出逆旋转在哪里执行或融合。
- 对 residual、RMSNorm/LayerNorm、非线性、RoPE、GQA 和分支逐项证明
  旋转合法性。
- 块级 `R_o^T Q(R_o W R_i^T) R_i` 重建只能称为 coefficient-domain
  probe，不能直接称为零开销模型旋转。

## 2. 变换族和公平基线

至少包含：

- identity；
- deterministic Hadamard；
- 多个 randomized Hadamard 种子，并报告 median/best-of-N；
- real DCT；
- FFT 时采用 Hermitian 独立自由度；
- learned rotation 使用独立 train/validation/test；
- SVD/Haar 等高自由度变换标为 unpriced oracle。

学习方法必须与随机搜索使用相同的数据和选择预算，报告角度数量、精度、
共享范围和 validation selection。

## 3. 实际 payload

必须计入：

- quantized values；
- per-group/per-block scales 和 zero point；
- sparse indices、mask 或 structured keep count；
- permutation、sign、phase、angle和 codebook；
- FFT complex/Hermitian 表示；
- transform 临时 buffer 和 inverse buffer。

SVD、full Haar 或 unconstrained `n×n` learned matrix 若不计两侧矩阵，
只能作为未计价上界。

## 4. 指标

系数级：

- top-k energy；
- structured low-frequency energy；
- PAPR、max/RMS、kurtosis；
- reconstruction relative MSE；
- bits/weight。

激活级：

- `E||(W-\hat W)x||² / E||Wx||²`；
- calibration/test token 严格分离；
- 分层、分投影类型报告。

模型级：

- perplexity 和下游任务；
- W/A/KV 位宽；
- prefill/decode 延迟、吞吐和峰值内存；
- transform 开销和是否真实融合。

## 5. FFT 特殊要求

- 实输入使用 Hermitian 共轭约束。
- 稀疏选择必须成对保留共轭项。
- complex coefficient 不得按一个 real value 计费。
- phase-only 实验必须明确 magnitude、self-conjugate 项是否精确保留。
- 报告 FFT/DCT 与矩阵乘法之间的数据布局和 kernel 成本。

## 6. Kuramoto/相位特殊要求

- 区分 KoPE 动态 token phase、FFT coefficient phase 和静态通道旋转。
- 旋转映射必须独立于全局 phase gauge。
- 纯吸引同步需先检查是否退化为全同相和恒等相对旋转。
- cluster synchronization 必须说明 signed/repulsive coupling、自然频率
  或簇约束。
- 只有在包含 phase/codebook 元数据后仍改善 rate-distortion，才可称为
  压缩收益。

## 7. 证据等级

- `sanity_synthetic`：只验证机制和实现。
- `real_weight_block`：真实权重块重建，不含模型任务。
- `activation_calibrated`：真实激活代理或 calibration capture。
- `model_quality`：完整模型 perplexity/任务。
- `deployable`：实际 payload、融合 kernel、延迟和内存全部验证。

结论不得跨级外推。
