# FreqKV 历史结果重评与真实 KV 初步验证

日期：2026-07-17

## 结论先行

对旧 `FFTNet/KV_FFT` 的准确判断不是“所有结果都是占位符”，而是：

1. `simulate_freqkv_compression` 中的 `k_energy_ratio=0.90` 和
   `v_energy_ratio=0.85` 是源码明确标注的硬编码 `Placeholder`，不是
   测量值。
2. `freqkv_final_results` 下的 15 张 PNG 和进度日志确实来自一次
   2025-05-16 的程序运行，不应称为伪造或占位图片。
3. 该次运行只使用独立高斯 K/V 和 RoPE-like 逐位置正交旋转，并实际
   走了 rFFT fallback；其图像显示接近平坦频谱，反而不支持“低频集中”。
4. rFFT 逆变换前把 complex 系数强制转成 float，丢弃虚部，解释了
   历史压缩图中即使保留约 90% 频率，报告能量仍只有约 45% 的现象。
5. 旧目录中没有找到真实模型 KV capture、perplexity、
   LongBench/RULER/NIAH、常驻 payload 字节数或端到端延迟结果。
6. 本次新增独立的真实模型探针：在 Qwen2.5-1.5B 全部 28 层、4 个
   WikiText-2 片段上捕获 RoPE 前 K、RoPE 后缓存 K 和缓存 V。原始
   RoPE 后 K 的 DCT 低频 25% 承载 `85.49%` 能量；逐头逐通道去掉
   序列均值后仍有 `65.51%`，明显高于白噪声的约 `25%`。V 的对应值
   为 `52.69%` 和 `43.97%`，结构更弱。

因此应停止传播旧实现的“4×、90%/85%、低损失”结论，但不能据此断言
所有真实模型 KV 的频域压缩都不可能成立。新增结果把判断更新为：
**真实 K 确有值得继续验证的频谱集中，V 较弱且层间差异明显；目前仍
只是频谱证据，不是压缩后模型质量、内存或速度证据。**

## 一、证据分级

| 内容 | 状态 | 可支持的结论 |
|---|---|---|
| `0.90/0.85` energy metrics | 源码硬编码占位符 | 只能证明曾计划计算该指标 |
| 15 张 `freqkv_final_results/*.png` | 真实执行产物 | 证明合成分析脚本运行过 |
| `analysis_log_20250516_190458.txt` | 真实进度日志 | 证明各绘图步骤完成，不含数值 |
| `freqkv_analysis_summary.md` | 运行后撰写的总结 | 结论与图中数据矛盾，不能独立采信 |
| 本次正确 DCT/rFFT 五种子重评 | `rerun_test` | 可否定旧合成数据的低频集中解释 |
| 本次 Qwen2.5-1.5B 真实 KV 频谱探针 | `rerun_test` | 支持真实 K/V 存在非均匀频谱，但不支持任务质量或部署收益 |

历史文件及 15 张图片的 SHA256 见
[`../provenance/freqkv_historical_artifacts_20260717.sha256`](../provenance/freqkv_historical_artifacts_20260717.sha256)，
分类清单见
[`../evidence/freqkv_historical_artifact_inventory_20260717.csv`](../evidence/freqkv_historical_artifact_inventory_20260717.csv)。

需要保留一个溯源限制：顶层 FreqKV 目录不是独立 Git 仓库，当前
`freqkv_example.py` 的 mtime 为 2025-09-17，晚于 2025-05-16 的历史
图片，因此无法恢复当日逐字一致的完整源码快照。不过，5 月版本的
`freqkv_final_analysis.py` 已明确写出“frequency domain size after
RFFT”，图片的 17/33/65 个频率点也与 seq32/64/128 的 rFFT 长度完全
一致。当前留存源码中的 `0.90/0.85` 明确是占位值，同时没有任何历史
数值文件可以把这两个数字认证为测量结果。

## 二、为什么历史合成数据不能证明低频集中

旧生成器先独立采样高斯 K/V，再只对 K 的每个相邻特征对施加
position-dependent 旋转。对每个位置和特征对，这都是二维正交变换：

\[
\begin{bmatrix}k'_{2j}\\k'_{2j+1}\end{bmatrix}
=
\begin{bmatrix}\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta\end{bmatrix}
\begin{bmatrix}k_{2j}\\k_{2j+1}\end{bmatrix}.
\]

独立各向同性高斯分布在正交变换后分布不变。它不会因为带有
“RoPE-like”循环就变成沿序列平滑的低频信号；V 甚至没有旋转。

本次保持旧数据模型，使用五个种子、序列长度 32/64/128、4 heads、
head dimension 64，分别测量：

- 正交 DCT-II 的低频截断；
- 保留完整 complex 系数并按 Parseval endpoint 权重计能量的 rFFT；
- 复现旧实现“丢弃虚部后 irFFT”的路径；
- 显式按 DCT 频率指数衰减构造的平滑正对照。

完整实现见
[`../../scripts/run_freqkv_reassessment.py`](../../scripts/run_freqkv_reassessment.py)
和
[`../../src/fft_com/freqkv_audit.py`](../../src/fft_com/freqkv_audit.py)。

## 三、正确 DCT 结果：能量基本跟随保留比例

seq64 五种子均值：

| 低频保留比例 | K 能量 mean ± std | V 能量 mean ± std |
|---:|---:|---:|
| 25% | `0.25067 ± 0.00338` | `0.24762 ± 0.00247` |
| 50% | `0.49864 ± 0.00390` | `0.50103 ± 0.00361` |
| 75% | `0.75214 ± 0.00149` | `0.75051 ± 0.00604` |

该规律在 seq32 和 seq128 上一致。例如 K 的前 25% DCT 能量分别为
`0.25462`、`0.25067`、`0.24934`。这与白噪声频谱的预期一致，不存在
“前 25% 系数承载 90% 能量”的信号。

若要求 90% DCT 能量，K 平均需要的真实分量比例为：

| 序列长度 | 所需 DCT 分量比例 |
|---:|---:|
| 32 | `90.63%` |
| 64 | `90.63%` |
| 128 | `90.47%` |

聚合结果见
[`../tables/freqkv_reassessment_20260717.csv`](../tables/freqkv_reassessment_20260717.csv)
和
[`../tables/freqkv_frequency_thresholds_20260717.csv`](../tables/freqkv_frequency_thresholds_20260717.csv)。

## 四、历史 rFFT 图为何只有约一半能量

旧路径实际执行：

1. `rfft` 得到 complex 系数；
2. 保留前若干频率；
3. 在 `irfft` 前执行 `tensor_freq.float()`；
4. complex 虚部被丢弃；
5. 再次 rFFT，并用未加 Parseval endpoint 权重的频域平方和报告能量。

seq64 K 的五种子复现：

| 请求保留比例 | 实际 rFFT bin 比例 | 截断前可保留的正确能量 | 历史口径报告能量 | 重建相对 MSE |
|---:|---:|---:|---:|---:|
| 25% | `24.24%` | `23.61%` | `13.99%` | `87.18%` |
| 50% | `48.48%` | `48.38%` | `25.88%` | `74.91%` |
| 90% | `87.88%` | `89.00%` | `45.46%` | `54.71%` |

因此历史图并非在展示一个“保留低频仍能很好重建”的结果；它展示的是
白噪声低频截断再叠加丢虚部错误后的高失真结果。

## 五、历史频谱图还有一个零基索引偏差

历史绘图代码使用：

```python
idx = np.searchsorted(cumulative_energy, threshold)
ratio = idx / len(cumulative_energy)
```

`idx` 是零基下标，达到阈值所需的真实分量数应为 `idx + 1`。因此图片
图例应按如下方式解释：

| 序列长度 | 阈值 | PNG 图例 | 同一 index 的真实所需比例 |
|---:|---:|---:|---:|
| 32 | 90% | `88.2%` | `94.1%` |
| 32 | 95% | `94.1%` | `100%` |
| 32 | 99% | `94.1%` | `100%` |
| 64 | 90% | `87.9%` | `90.9%` |
| 64 | 95% | `93.9%` | `97.0%` |
| 64 | 99% | `97.0%` | `100%` |
| 128 | 90% | `89.2%` | `90.8%` |
| 128 | 95% | `93.8%` | `95.4%` |
| 128 | 99% | `98.5%` | `100%` |

这个偏差不改变主要判断：频谱仍接近平坦，但历史图例进一步低估了达到
目标能量所需的频率比例。

## 六、正对照证明评测能够识别真正低频结构

对显式按 DCT 频率指数衰减构造的平滑 K，低频 25% 能量为：

| 序列长度 | 低频 25% 能量 |
|---:|---:|
| 32 | `95.83%` |
| 64 | `99.82%` |
| 128 | `99.9997%` |

seq64 保留 50% 后能量为 `99.9997%`。这说明正确实现并非机械地产生
“能量等于保留比例”的结果：输入真正平滑时，它能明确测出低频集中。

## 七、真实模型 KV：出现了值得继续验证的 K 频谱结构

本次新增
[`../../scripts/run_freqkv_real_kv_probe.py`](../../scripts/run_freqkv_real_kv_probe.py)，
不复用旧 FreqKV 实现。实验使用：

- Qwen2.5-1.5B，28 层、12 个 query heads、2 个 KV heads；
- 全部层 `0..27`；
- WikiText-2 raw test 的 4 个互不重叠 512-token 片段，token offsets 为
  `0/4096/8192/12288`；
- 每个片段的位置编号重新从 `0..511` 开始，数据 token offset 不作为
  RoPE position offset；
- FP16，在 RTX 4090 上捕获 `K_pre_rope`、`K_post_rope` 和 `V_cache`；
- DCT-II 与按 Parseval 权重计算的 rFFT，各报告低频前缀和任意频率
  top-bin oracle。

每个 signal 的总体统计包含 28 层 × 4 片段，即 112 个样本。DCT
低频 25% 的结果如下：

| signal | 原始 DC 能量 | 原始低频 25% | 去均值低频 25% | 去均值 top-25% oracle |
|---|---:|---:|---:|---:|
| K，RoPE 前 | `71.02%` | `86.82%` | `52.28%` | `56.12%` |
| K，RoPE 后 | `57.76%` | `85.49%` | `65.51%` | `70.01%` |
| V cache | `15.76%` | `52.69%` | `43.97%` | `46.95%` |

这里必须同时看原始和去均值结果。原始 K 的高能量集中很大一部分来自
序列 DC，即每个 head/channel 的序列均值；一个 DC 系数可以精确表示
这部分，但在自回归更新中仍需计算其增量维护和真实 payload 成本。
去均值后，RoPE 后 K 的低频 25% 仍承载 `65.51%` 能量，显著高于
白噪声预期的约 `25%`，且只比任意频率 top-25% oracle 低 `4.50`
个百分点。这说明结构主要位于低频前缀，而不只是少数散落频点。

但它还不是“4× 且近乎无损”。对去均值信号保留 90% DCT 能量，平均
需要的低频分量比例为：

| signal | 达到 90% 能量所需比例 |
|---|---:|
| K，RoPE 前 | `75.55%` |
| K，RoPE 后 | `65.05%` |
| V cache | `81.36%` |

层间差异同样明显。去均值低频 25% 的全层分布为：

| signal | 最低层 | 中位数 | 最高层 | ≥60% 的层数 |
|---|---:|---:|---:|---:|
| K，RoPE 前 | `30.95%`（L0） | `53.69%` | `68.53%`（L16） | `6/28` |
| K，RoPE 后 | `55.51%`（L25） | `65.81%` | `74.10%`（L19） | `23/28` |
| V cache | `27.01%`（L1） | `45.80%` | `60.26%`（L16） | `1/28` |

因此后续不应对所有层和 K/V 使用同一截断比例。K 比 V 更值得优先，
中间层通常比首尾层更有潜力。DCT 与 rFFT 的总体判断一致：去均值
RoPE 后 K 的低频 25% 能量分别为 `65.51%` 和 `65.30%`，说明结果
不是某个变换定义偶然造成的。

RoPE 前后 K 的总能量比在全部样本中位于
`0.999931..1.000111`，符合旋转保持范数；原始低频 25% 从
`86.82%` 变为 `85.49%`。去均值后 RoPE 后 K 反而更集中，说明
position-dependent 旋转会改变序列频率分布，不能只凭范数守恒推断
频谱形状。

同一脚本分别在 `CUDA_VISIBLE_DEVICES=0` 和 `=1` 上运行，7 个生成
CSV 的 SHA256 全部一致。模型、数据、脚本和发布结果哈希见
[`../provenance/freqkv_real_kv_20260717.sha256`](../provenance/freqkv_real_kv_20260717.sha256)；
机器可读结果见
[`../evidence/freqkv_real_kv_summary_20260717.json`](../evidence/freqkv_real_kv_summary_20260717.json)、
[`../tables/freqkv_real_kv_results_20260717.csv`](../tables/freqkv_real_kv_results_20260717.csv)
和
[`../tables/freqkv_real_kv_thresholds_20260717.csv`](../tables/freqkv_real_kv_thresholds_20260717.csv)。

本实验没有把截断后的系数注回 attention，也没有测 perplexity、
生成质量、长上下文任务、真实常驻字节数或延迟。它支持的是
“优先继续做真实 post-RoPE K、DC/AC 分离、逐层自适应的压缩验证”，
而不是恢复旧 FreqKV 的宣传数字。

## 八、迭代压缩和内存图的证据边界

历史 sawtooth/token-count 图可以证明脚本中的计数器会周期性压缩，但
不能证明真实模型显存下降：

- `process_new_kv_states` 会重复追加当前 token；
- 返回值包含原始 K/V、完整 FFT、全尺寸零填充副本和重建张量；
- 没有紧凑系数 payload、索引序列化或常驻字节数；
- “compressed size”只记录频率 bin 数，没有计入 complex 实部/虚部、
  dtype、sink/recent token 和临时 buffer；
- 真实模型测试加载失败会自动回退到 demo；
- 手写生成循环没有正确把 `past_key_values` 回传给后续 token。

所以这些图保留为历史执行证据，不进入内存、速度或质量结论。

## 九、更新后的研究判断

### 建议继续

- 真实 post-RoPE K 的 DC/AC 分离和逐层自适应截断；
- 对 K 与 V 使用不同策略，优先验证 K，不强求统一压缩比例；
- 在 Llama/Qwen 等不同架构上复核层间频谱差异；
- 历史 PNG、日志、源码和本次反证表；
- DCT/rFFT 的正确数值检查和低频正对照；
- 与 R-KV、ShadowKV 等方法建立统一比较协议的需求。

### 应撤回

- 旧 README 的 4× 压缩且信息损失很小；
- K/V 能量保持 90%/85%；
- RoPE-like 高斯样本证明低频集中；
- token-count sawtooth 等价于真实显存收益；
- 历史原型已经在 LLM 上有效。

### 下一阶段最低实验要求

1. 把截断后的 K/V 真正注回 attention，先测 attention-output 误差和
   WikiText-2 perplexity；
2. 明确 DC/AC 的在线更新方式、变换轴、RoPE 位置、GQA/head 边界和
   sink/recent 策略；
3. 保存真正紧凑的 payload，而不是全尺寸零填充频谱；
4. 做逐层、逐 K/V 的 rate allocation，并与统一比例消融；
5. 与 full KV、token eviction、R-KV、ShadowKV 使用同一任务和预算；
6. 同时报 perplexity/长上下文任务、常驻及峰值字节数、prefill/decode
   延迟和失败案例；
7. 固定随机种子、环境、模型版本、数据 split 和结果 JSON/CSV。

本次最强结论是：

> 旧 FreqKV 的两项宣传数字确为占位符；历史图片是真实合成运行，但
> 数值和实现均不支持其乐观总结。独立真实模型探针表明 post-RoPE K
> 存在可继续验证的低频结构，但 V 较弱且层间差异大；下一步必须用
> 真正压缩后的 cache、模型质量和实存字节数决定它能否成为方法。
