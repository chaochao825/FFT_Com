# 研究线划分与继续价值判断

日期：2026-07-16
范围：服务器 34 的 `/data2/wangmeiqi`，重点为 FFT、KV 频域压缩和
`learnable_logic_hadamard_mixer_20260716`。

## 数值验证补充

本文件前半部分记录整理时对旧原型的判断。此后 FFT_Com 新增了独立、
不复用旧 FreqKV/fourier_trans 实现的 NumPy 评测框架，并在真实
Llama-2-7B 权重上完成五种子块级验证。

新证据把“频域方向整体暂停”细化为：

- 继续 Hadamard/FFT/DCT 作为量化前正交基选择的模型级验证；
- 停止原始 channel order 上的 DCT/FFT 低频裁剪；
- 暂停当前 DCT sparse base + Hadamard residual；
- 停止朴素纯吸引 Kuramoto 离线旋转；
- learned rotation 只有超过 best-of-N random 的留出集结果后才升级。

详见
[`transform_potential_study_20260716.md`](transform_potential_study_20260716.md)。
旧目录的实现和宣传结论仍不因此恢复有效。

## 一、目录应分成四部分

### A. 当前主线：离散逻辑与可部署交易路径

源目录：
`/data2/wangmeiqi/learnable_logic_hadamard_mixer_20260716`

这部分值得继续。稳定价值不是一个单独的“新变换”，而是把训练态
fake-quant 模型逐步收敛到明确的整数/位级部署契约：

- Wmag7/A8 编码与精确 A8×U4 LUT 交易；
- 符号和幅值位平面的可解释重构；
- XNOR/popcount 相似度和固定 tie-break 的 hard Top-K；
- Q0.15 RMS-LUT、Shift-RMS、requant、no-norm 对照；
- 去除 FP32 shadow、优化器和 STE 对象的部署导出；
- 116 个 CPU 测试覆盖核心模块、导出器、协议哈希和交易后端。

当前证据并不支持“Hadamard/LHVM 是更优全局混合器”。1k-step 的
Hadamard/LHVM smoke 均低于 attention 对照；正在运行的 ScaleLogic
50k 实验也仍以 attention 为全局混合器，只在前四层使用 depthwise
shift-add。

### B. 自己的频域探索：保留证据，但不沿原实现继续堆补丁

包括：

- `FFTNet` 根目录的 LLM 权重 FFT 笔记本与工具；
- `FFTNet/KV_FFT` 顶层的 FreqKV 原型；
- `FFTNet/Freq_KV` 的轻量说明/图片副本；
- `FFTNet/fourier_trans` 的傅里叶点积探索；
- `/data2/wangmeiqi/fft/order_test` 对上游仓库的本地未跟踪增量。

这些目录有探索价值，但当前代码和证据不能支持压缩收益。建议将它们
作为“失败原型与比较需求”保留，后续若重启频域方向，应从统一协议和
正确的 compact payload 开始，而不是继续在旧脚本上修补。

### C. 已有工作：第三方上游快照

以下目录是外部 Git 仓库，不应作为自己的源码合并发布：

- `jacobfa/fft`
- `Zefan-Cai/R-KV`
- `ByteDance-Seed/ShadowKV`
- `cjyaras/monarch-attention`

本仓库只记录 URL、提交号、本地修改状态和比较角色。具体信息见
[`../provenance/source_inventory_20260716.csv`](../provenance/source_inventory_20260716.csv)
与 [`../../comparisons/README.md`](../../comparisons/README.md)。

### D. 同一大目录中的非本课题内容

`FFTNet/non_linear`、`flashSVD`、`ngc-learn`、
`plasuiable_learning` 等与本次 FFT/KV/离散逻辑比较不是同一问题。
它们不进入 `FFT_Com`，也不因目录名相邻而被视作频域压缩证据。

## 二、继续、暂停和停止

### 立即继续

1. 完成 d12/e384 ScaleLogic `local_layers=4` 与 `local_layers=0`
   的冻结协议配对实验。
2. 对胜出设置至少运行 3 个种子，报告均值、标准差和失败运行。
3. 完成 accumulator/residual 到 A8 的端到端整数 exponent-only
   requant 边界，并逐层核对 fake-quant、交易参考和导出路径。
4. 所有长实验持久化 JSONL、protocol、源码哈希和独立最终结果。
5. 在同一训练预算下再加入 attention、Hadamard/LHVM 和频域基线。

### 暂停，等待匹配证据

- Hadamard/LHVM 的精度或效率优势；
- FreqKV 的 4× 内存收益和低损失结论；
- 仅由频谱图片推导出的压缩结论；
- 没有 compact payload 的“压缩后张量”内存比较；
- 没有真实模型任务指标的 KV cache 方法比较。

### 停止沿旧实现继续

- `fourier_trans` 当前的错误点积公式；
- `order_test` 当前的覆盖式“shuffle”；
- FreqKV 当前重复 token 的 cache 更新逻辑；
- 把硬编码的 0.90/0.85 占位值当作测量结果；
- 将 1 字节空笔记本继续作为项目入口。

## 三、为什么两条线不能直接比较

频域 KV 压缩主要改变推理时状态表示；full-discrete/ScaleLogic 当前
主要改变模型数值、路由和部署交易。二者回答的问题不同。

只有在以下条件完全匹配时，才可以把结果放进同一结论表：

- 同一模型、数据、split、种子和训练/校准预算；
- 明确压缩对象、保留比例、实存字节数和还原路径；
- 同一硬化/部署状态下的任务质量；
- 参数、KV、激活、临时 FFT buffer 和索引元数据全部计入；
- 延迟、吞吐、能耗或综合结果来自同一硬件和 batch/context；
- 失败、OOM 和不兼容运行同样记录。

统一协议见
[`../protocols/matched_comparison_protocol.md`](../protocols/matched_comparison_protocol.md)。

## 四、整理结果

- 主线源码和证据已同步到 `chaochao825/learnable_logic`。
- `FFT_Com` 保存判断、来源、轻量结果和下一轮统一协议。
- 外部源码没有复制进本仓库。
- 大 checkpoint 和图片批次仍留在服务器原始区。
- 明确的空文件、缓存、空结果目录和字节级重复运行已移入可恢复的
  `trash/`，清单记录在 `docs/provenance/cleanup_staging_20260716.tsv`。
