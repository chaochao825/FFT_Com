# 正交变换与 FFT_Com 潜力验证

日期：2026-07-16

## 结论先行

本轮把 Hadamard、DCT、FFT、受约束学习旋转和 Kuramoto 型相位方案放在
同一套块级协议下验证后，结论不是“频域压缩成立”，而是更窄、更可执行：

1. **值得继续的是量化友好的正交基选择**。在真实 Llama-2-7B Q 投影
   权重块上，Hadamard 和 Hermitian FFT 的 3-bit 权重重建误差均稳定
   低于原域量化；用真实 layer-0 token embedding 构造的输入代理上，
   FFT/Hadamard 的输出误差也明显降低。
2. **原始通道顺序上没有可用的 DCT 低频稀疏性**。DCT 的 top-k 能量
   反而低于原权重，低频保留比例与能量比例基本相同。
3. **当前“DCT 稀疏基底 + Hadamard 低比特残差”不适合 Llama 权重**。
   它在人工平滑矩阵上非常有效，但在真实权重上基底没有拿走足够能量，
   残差仍接近完整权重，rate-distortion 明显输给直接量化。
4. **受约束学习旋转存在训练集收益，但没有超过随机 Hadamard 的留出集
   证据**。当前代理出现过拟合，不能据此声称 SpinQuant 型学习有效。
5. **朴素纯吸引 Kuramoto 离线旋转退化**。同步后相对相位趋近零，
   gauge-invariant 配对旋转回到恒等；FFT 相位跨块也没有足够集中度。

因此，下一阶段应从“频率稀疏压缩”转向“合法模型旋转 + 激活感知量化”，
并把 DCT 稀疏混合和朴素 Kuramoto 暂停。

## 一、已有工作、自己的设想和本轮证据

| 类别 | 内容 | 在本研究中的角色 |
|---|---|---|
| 已有工作 | [QuaRot](https://arxiv.org/abs/2404.00456) | 用计算友好的 Hadamard 旋转消除量化离群值，是随机旋转基线 |
| 已有工作 | [SpinQuant](https://arxiv.org/abs/2405.16406) | 说明不同随机旋转质量差异很大，并学习功能等价的量化旋转 |
| 已有工作 | [FlatQuant](https://arxiv.org/abs/2410.09426) | 强调 layer-specific 变换、运行开销和 Kronecker 低成本实现 |
| 已有工作 | [KoPE 论文](https://arxiv.org/abs/2604.07904) 与 [官方代码](https://github.com/microsoft/Neuro-inspired_Phase_Encoding/blob/main/vit_kope.py) | 动态 token 相位状态与 Kuramoto 式更新；不是离线权重压缩 |
| 自己的设想 | Hadamard、FFT/DCT、可学习 Givens、Kuramoto 相位统一为正交变换注册表 | 本轮需要验证的研究假设 |
| 自己的设想 | DCT/FFT 稀疏基底 + Hadamard 低比特残差 | 本轮重点 rate-distortion 候选 |
| 历史本地尝试 | FFT 权重图、FreqKV、fourier_trans、order_test | 已审计的探索或失败原型，不作为本轮正证据 |
| 本轮新证据 | `src/fft_com/`、11 个单元测试、真实 Llama 块实验和五种子表 | FFT_Com 自己的可复现验证 |

KoPE 的官方实现维护 `[B,H,N,D/2]` 的 `phase_cos/phase_sin` 状态，用
token 特征形成 attention-style coupling，将更新方向投影到当前相位的
切空间后重新归一化，并用相位旋转 Q/K/V。它支持“同步可以成为神经网络
动态机制”，但不支持“对静态权重运行一次 Kuramoto 就会更可压缩”。

## 二、统一框架中必须补上的数学边界

### 1. 两侧正交变换不是免费操作

块级分析使用

\[
C = R_o W R_i^\mathsf{T},\qquad
\hat W = R_o^\mathsf{T}Q(C)R_i.
\]

这能严格测量系数量化后的理想重建误差，但模型推理若直接使用 `C`，还
必须在输入侧执行 `R_i`、输出侧执行 `R_o^T`，或证明它们可以合法融合
到相邻线性层。残差、RMSNorm/LayerNorm、非线性和分支都可能阻止融合。
因此本轮结果是“变换域潜力”，不是零开销模型部署结果。

### 2. FFT 必须按 Hermitian 独立自由度计费

实矩阵的二维 FFT 是复数并满足共轭对称。本实现没有把一个 complex 当成
一个 real：它显式枚举共轭组，恰好保存 `n²` 个独立实标量，稀疏选择也
按共轭组和实标量预算执行。FFT 仍有复数乘加、临时 buffer 和 kernel
成本，后续部署必须另外计入。

### 3. 任意两侧正交旋转会把“稀疏性”问题平凡化

如果不限制变换和元数据，SVD 可令 `U^T W V` 只有至多 `n` 个非零对角
项。64×64 块中，top-1/64 系数预算已经可以保存全部 64 个奇异值，本轮
SVD oracle 的能量保持因此是 100%。这不是可部署压缩收益；`U/V` 本身
各有 `n²` 参数和 `O(n²)` 计算。学习旋转比较必须限制结构、共享范围和
元数据，并把 SVD 只当作未计价上界。

### 4. “相位旋转”需要区分三件事

- FFT 的固定复相位基；
- 实数 Givens/SpinQuant 类通道旋转；
- KoPE 的动态 token phase state。

它们可以用 unitary/orthogonal 语言联系，但计算图、状态和可融合条件
不同，不能只因都出现 `cos/sin` 就视为同一算法。

## 三、实验协议

- 模型：本地 `Llama-2-7b-chat-hf` safetensors。
- 层：0、8、16、24、31。
- 权重：`self_attn.q_proj.weight` 和 `mlp.down_proj.weight`。
- 主分析：每类每层 8 个对齐的 64×64 块，共 80 个真实权重块/种子。
- 稳健性：采样种子 `20260716` 至 `20260720`，共五次完整运行。
- 合成 sanity：平滑 DCT 生成矩阵、稀疏大离群值矩阵、Gaussian 矩阵。
- 变换：identity、DCT-II、Hadamard、随机 Hadamard、随机 Haar、
  Hermitian FFT，以及未计价 SVD oracle。
- 量化：每块 abs-max 对称 3/4-bit；2-bit 残差为有零点的三值对称量化。
- 激活代理：随机抽取 512 个真实 token embedding，执行第 0 层
  input RMSNorm，再测 sampled Q 子块的线性输出误差。
- 学习旋转：32×32、五级 real butterfly Givens，共享 160 个角度；
  24 个训练块和 24 个独立留出块。
- 运行：NumPy CPU，`CUDA_VISIBLE_DEVICES` 为空；没有占用活动训练 GPU。

全部代码和命令见仓库根目录 README。正式结果：

- `docs/tables/transform_metrics_20260716.csv`
- `docs/tables/compression_rate_distortion_20260716.csv`
- `docs/tables/transform_seed_sweep_20260716.csv`
- `docs/tables/learned_rotation_20260716.csv`
- `docs/tables/kuramoto_phase_probe_20260716.csv`

## 四、机制 sanity：实现本身能区分不同现象

### 平滑矩阵：DCT 稀疏成立，但 dense 低比特量化未必成立

人工在 DCT 域按频率指数衰减生成的矩阵中：

- DCT zigzag 前 12.5% 系数能量：`0.999989`；
- DCT + Hadamard 2-bit 残差，约 3.02 bpp：相对 MSE `0.000400`；
- 原域 3-bit：相对 MSE `0.08938`。

这说明混合方案在“确实有平滑低频结构”时可以工作。同时，DCT dense
3-bit 的误差为 `0.1590`，反而高于原域 3-bit：能量集中会增大峰值，
并不自动改善统一 scale 的 dense quantization。

### 离群矩阵：PAPR 下降也不等于权重量化误差下降

人工稀疏大离群值矩阵中，Hadamard 将平均 PAPR 从 `1631.4` 降到
`6.95`，但 3-bit 相对 MSE 从 `0.03985` 升到 `0.06337`。原域量化可以
精确保留少数大值、牺牲小噪声；Hadamard 把大值摊到全部系数后，反而
产生更多量化误差。

因此 QuaRot/SpinQuant 的收益不能简化成“权重 PAPR 越低越好”，必须放
回完整网络、激活和功能等价旋转中验证。

### Gaussian：没有凭空产生结构

Gaussian 矩阵各实正交变换的 top-k 能量和量化误差接近，证明代码没有
对 DCT/Hadamard 人为制造普遍优势。

## 五、真实 Llama 权重结果

### 1. 不支持 DCT 频率稀疏

五种子中，Q 投影 DCT top-12.5% 能量与原域 top-12.5% 的比值为：

\[
0.8605 \pm 0.0077.
\]

代表种子中，原域为 `0.6095`，DCT 为 `0.5198`。更关键的是，DCT
zigzag 低频 12.5% 系数只保留 `0.1295` 能量，几乎等于随机均匀分布的
12.5%。因此“相邻 channel index 具有自然频率平滑性”在这些权重上不
成立。

### 2. 支持量化前正交平坦化

五种子相对原域 3-bit 的平均误差比：

| 对象/方法 | 相对误差比 mean ± std | 判断 |
|---|---:|---|
| Q projection / Hadamard | `0.619 ± 0.029` | 稳定改善 |
| Q projection / FFT | `0.605 ± 0.034` | 稳定改善 |
| down projection / Hadamard | `0.786 ± 0.047` | 改善较弱 |

代表种子 Q 投影的绝对块级相对 MSE：

| 约 3.01 bpp 方法 | 权重相对 MSE | layer-0 激活代理输出 MSE |
|---|---:|---:|
| 原域 3-bit | 0.2442 | 0.08651 |
| DCT 3-bit | 0.1593 | 0.03361 |
| Hadamard 3-bit | 0.1438 | 0.02890 |
| FFT 3-bit | 0.1498 | 0.01797 |

五种子中，FFT 3-bit 激活代理误差是原域 3-bit 的
`0.329 ± 0.080`，Hadamard 为 `0.459 ± 0.085`。这是本轮最值得继续
验证的正结果。

但 FFT 3-bit 相对原域 4-bit 激活误差的比值为
`1.704 ± 0.479`，只有代表种子接近 1；不能宣传为“稳定达到 4-bit
质量”。此外当前只测了第 0 层 sampled sub-block，不是完整 Q 矩阵或
模型 perplexity。

### 3. DCT 基底 + Hadamard 残差没有真实权重优势

五种子：

- 12.5% DCT zigzag Q8 基底 + Hadamard 2-bit 残差，约 3.02 bpp，
  误差是纯 Hadamard 3-bit 的 `4.876 ± 0.125` 倍。
- 同一基底 + Hadamard 3-bit 残差，约 4.02 bpp，误差是原域 4-bit 的
  `2.280 ± 0.084` 倍。

代表种子中，3-bit 残差版本 Q 权重 MSE 为 `0.1340`，虽略低于纯
Hadamard 3-bit 的 `0.1438`，但付出了约 1 bit/weight，仍远差于原域
4-bit 的 `0.0605`。当前通道顺序没有低频结构，稀疏基底没有形成值得
支付的额外 payload。

## 六、学习旋转：有训练收益，但未通过留出集

代表种子的 32×32 Q 子块：

| 方法 | 训练 MSE | 留出 MSE |
|---|---:|---:|
| identity | 0.1710 | 0.1525 |
| Hadamard-like butterfly | 0.1240 | 0.1200 |
| best-of-16 randomized Hadamard | 0.1168 | 0.1137 |
| 学习 butterfly | **0.08284** | 0.1292 |

学习过程显著降低训练误差，但留出集比 best randomized Hadamard 高
13.6%。五种子学习/随机 Hadamard 留出误差比为：

\[
1.125 \pm 0.056.
\]

因此 H2“学习旋转优于随机 Hadamard”在当前代理上被否定。更合理的下一
版应使用：

- 更大、跨层的训练/验证块集合；
- activation-weighted 而非纯权重 abs-max MSE；
- Hadamard 初始化附近的角度正则；
- validation early stopping；
- 与 best-of-N random 使用相同选择预算。

这不是对 SpinQuant 的复现实验，也不能据此否定其完整模型结果。

## 七、Kuramoto 与 FFT 相位

### 1. 朴素离线旋转退化为恒等

对行/列绝对相关图运行零自然频率、纯吸引 Kuramoto：

- 平均 order parameter：`0.1620 → 1.0000`；
- 同步后 gauge-invariant 配对相位角 RMS：`1.07e-15`；
- 3-bit 误差相对 identity：`1.0000`。

这不是数值失败，而是当前构造的必然趋势：全部 oscillator 同步时，
相对相位消失；若使用绝对共同相位构造旋转，结果又依赖任意 gauge。
要得到簇旋转，必须引入有明确压缩目标的 signed/repulsive coupling、
自然频率或多簇约束。

### 2. FFT 相位共享信号很弱

40 个 Q 权重块的同频相位 magnitude-weighted circular concentration：

- 五种子 median：`0.202 ± 0.004`；
- 代表种子 mean/median/p90：`0.209 / 0.195 / 0.353`。

代表种子中，3-bit uniform phase-only MSE 为 `0.05130`，增加每频率
16-bit 模板后的 3-bit residual 为 `0.05099`，几乎没有变化；
全局 circular codebook 也只有千分量级改善。当前没有证据支持把
Kuramoto/聚类用于跨块 FFT phase sharing。

相位实验保留精确 magnitude 和 self-conjugate 系数，只隔离 phase
编码，因此也不是完整 FFT 压缩率。

## 八、继续、暂停和停止

### 立即继续：优先级 1

1. 在 Llama 的合法功能等价旋转位置复现 Hadamard、real DCT/FFT proxy
   和 learned rotation，确保残差、RMSNorm、RoPE、GQA 语义不变。
2. 使用真实 calibration token 收集多层激活，优化
   `E||(W-\hat W)x||²`，不要只优化权重 MSE。
3. 完整报告 W/A/KV bit-width、perplexity、下游任务、实际 payload、
   transform buffer 和 prefill/decode 延迟。
4. 先比较 Hadamard、best-of-N random 和受正则学习旋转；FFT 若进入
   模型路径，必须给出 real/Hermitian kernel 与融合方案。

### 条件继续：优先级 2

- 学习 butterfly/Givens：只有在留出集和完整模型上超过 best random
  才继续扩大。
- DCT sparse base：只有先通过通道重排、分组或 learned permutation
  获得显著低频能量集中，并计入重排元数据后再重启。
- Kuramoto phase clustering：只有定义独立于 gauge 的簇目标和实际
  rate-distortion 收益后再做；与 KoPE 动态网络实验分仓或分支处理。

### 当前停止

- 在原始 Llama channel order 上直接做 DCT/FFT 低频裁剪；
- 当前 DCT + 2/3-bit Hadamard residual 配方；
- 纯吸引、零自然频率 Kuramoto 后直接映射为离线权重旋转；
- 将 KoPE 的训练效率结果当作权重压缩证据。

## 九、证据边界

本轮已经把“图片观察”升级为可复现数值实验，但仍未完成：

- 完整矩阵、完整层或完整模型重建；
- perplexity、zero-shot/long-context 任务；
- activation/KV quantization；
- 合法融合后的 transform runtime；
- 真实压缩文件或部署 kernel。

因此当前最强结论是“Hadamard/FFT 值得进入下一轮模型级量化实验”，
不是“FFT_Com 已经实现有效模型压缩”。
