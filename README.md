# FFT_Com

这是服务器 34 上频域压缩、正交旋转、FFT 权重分析与离散逻辑主线的
证据化整理仓库。它不是旧目录的镜像，也不复制第三方仓库；目标是明确
区分：

1. 已有工作的本地快照；
2. 自己做过的探索、失败尝试和可复用证据；
3. 当前值得继续投入的主线；
4. 后续进行公平对比所需的统一协议。

初次整理：2026-07-16；模型级复核更新：2026-07-17。

## 2026-07-17 FreqKV 历史重评与真实 KV 初步验证

- **不是所有历史结果都是占位符**：`0.90/0.85` 两项 energy metric
  是源码明确标注的硬编码 `Placeholder`；但 2025-05-16 的 15 张 PNG
  和进度日志确实来自一次合成程序运行。
- **正确 DCT 复核不支持低频集中**：seq64、五种子下，低频 25% 对
  Gaussian/RoPE-like K/V 只保留 `25.07%/24.76%` 能量；低频 50%
  保留约 `49.86%/50.10%`。
- **历史图中的约一半能量来自实现错误**：rFFT fallback 在 irFFT 前
  丢弃虚部；seq64 K 请求保留 90% 时，实际保留 87.88% bins，正确
  截断能量约 89.00%，历史口径只报告 `45.46%`。
- **正对照通过**：显式平滑的 seq64 K 在低频 25% 中保留 `99.82%`
  能量，说明重评能够检测真正的低频结构。
- **真实 Qwen2.5-1.5B K 出现可继续验证的结构**：28 层 × 4 个
  WikiText-2 片段上，RoPE 后 K 的原始 DCT 低频 25% 能量为
  `85.49%`；逐头逐通道去序列均值后仍为 `65.51%`，而 top-25%
  频率 oracle 为 `70.01%`。
- **V 较弱且必须逐层处理**：V 的原始/去均值低频 25% 能量为
  `52.69%/43.97%`；去均值后各层 post-RoPE K 为
  `55.51%..74.10%`，V 为 `27.01%..60.26%`。
- **更新后的边界**：旧实现及其“4×、90%/85%、低损失”结论继续
  撤回；真实 K 的频谱方向升级为值得做压缩注入实验，但当前没有
  perplexity、生成质量、紧凑 payload、内存或延迟收益。

完整报告和机器可读结果见：

- [`docs/reports/freqkv_reassessment_20260717.md`](docs/reports/freqkv_reassessment_20260717.md)
- [`docs/tables/freqkv_reassessment_20260717.csv`](docs/tables/freqkv_reassessment_20260717.csv)
- [`docs/tables/freqkv_frequency_thresholds_20260717.csv`](docs/tables/freqkv_frequency_thresholds_20260717.csv)
- [`docs/evidence/freqkv_reassessment_summary_20260717.json`](docs/evidence/freqkv_reassessment_summary_20260717.json)
- [`docs/tables/freqkv_real_kv_results_20260717.csv`](docs/tables/freqkv_real_kv_results_20260717.csv)
- [`docs/tables/freqkv_real_kv_thresholds_20260717.csv`](docs/tables/freqkv_real_kv_thresholds_20260717.csv)
- [`docs/evidence/freqkv_real_kv_summary_20260717.json`](docs/evidence/freqkv_real_kv_summary_20260717.json)

## 2026-07-17 模型级复核

- **合法性 sanity check 通过**：不量化时，完整 `all_input` DCT 路径
  PPL 为 `6.195030`，同协议 FP16 基线为 `6.195575`，说明在线正交
  变换及逆变换没有破坏模型功能。
- **统一 all-input dense rotation 不成立**：q4 Identity/DCT PPL 为
  `6.5672/6.6778`；q3 为 `9.4400/40.2434`。DCT、Hadamard、RDFT
  虽降低权重 MSE，却恶化了模型级误差结构，不能用局部 MSE 代替 PPL。
- **本轮唯一可暂时保留的候选信号是 q_proj 输入侧 q3 旋转**：三个
  互不重叠的 8,192-token test 区段上，DCT 相对同 scope Identity 的平均
  `ΔPPL=-0.0237`，`3/3` 获胜；Hadamard 为 `-0.0244`，两者质量
  基本同档。输出侧 DCT 仅 `2/3` 获胜且波动更大。
- **Permutation、双侧旋转和 head/RoPE-aware 方案均不支持继续**：
  spectral permutation 没有稳定超过普通 DCT，还需最多约
  `1.47 Mbit` 元数据和在线 gather；head-aware 与 RoPE-pair DCT
  也都差于同 scope Identity。
- **DCT 是更快的实数 FFT proxy，但仍有明显在线成本**：在
  `[1,256,4096]`、group 128 的未融合 PyTorch 参考实现中，DCT 为
  `0.2714 ms`，Hadamard 为 `0.5356 ms`；但代表性 q_proj 路径 DCT
  仍超过 Identity 的 4 倍。当前结果不代表 packed INT3/INT4 kernel。

完整模型结论、协议与机器可读结果见：

- [`docs/reports/model_rotation_study_20260717.md`](docs/reports/model_rotation_study_20260717.md)
- [`docs/protocols/legal_model_rotation_protocol.md`](docs/protocols/legal_model_rotation_protocol.md)
- [`docs/tables/model_rotation_results_20260717.csv`](docs/tables/model_rotation_results_20260717.csv)
- [`docs/tables/segment_robustness_20260717.csv`](docs/tables/segment_robustness_20260717.csv)
- [`docs/tables/transform_latency_20260717.csv`](docs/tables/transform_latency_20260717.csv)

## 2026-07-16 块级结论

- **继续 full-discrete 主线**：`learnable_logic` 的 Wmag7/A8、位平面、
  XNOR/popcount、整数 LUT 交易和部署导出仍是独立主线。
- **新增值得继续的方向**：把 Hadamard、FFT/DCT 和受约束学习旋转作为
  “量化前正交基选择”，下一步必须进入合法的模型旋转、激活校准和任务
  指标，而不是只看权重频谱。
- **不支持原始通道顺序上的 DCT/FFT 稀疏主张**：五个采样种子中，
  Llama Q 投影经 DCT 后 top-12.5% 能量相对原权重平均只有
  `0.860 ± 0.008`；低频 12.5% 系数也只承载约 12.9% 能量。
- **Hadamard/FFT 的量化潜力是真实但有限的块级证据**：Q 投影 3-bit
  权重误差相对原域量化分别约为 `0.619 ± 0.029` 和
  `0.605 ± 0.034`；layer-0 嵌入激活代理上，FFT 3-bit 输出误差约为
  原域 3-bit 的 `0.329 ± 0.080`。这不是 perplexity 或端到端精度。
- **暂停当前混合方案**：DCT 低频基底 + Hadamard 2-bit 残差在真实
  Llama Q 权重上比纯 Hadamard 3-bit 差约 `4.88×`；3-bit 残差版本在
  约 4.02 bpp 下仍比原域 4-bit 差约 `2.28×`。
- **不沿朴素 Kuramoto 离线旋转继续**：纯吸引同步把相对相位收敛到
  零，旋转退化为恒等；FFT 相位跨块集中度也很低。KoPE 应作为动态
  token 相位机制单独研究，不能直接作为权重压缩证据。
- **旧 FreqKV/fourier_trans/order_test 不作为正证据**：FreqKV 的
  图片是真实合成运行，但关键宣传数值是占位符，且正确复核不支持其
  解释；旧实现停止。独立真实 KV 探针已经发现 K 的初步正信号，但
  尚未形成模型质量或部署结论。

详细依据见
[`docs/reports/transform_potential_study_20260716.md`](docs/reports/transform_potential_study_20260716.md)、
[`docs/reports/research_line_decision_20260716.md`](docs/reports/research_line_decision_20260716.md)
和
[`docs/reports/frequency_prototypes_audit_20260716.md`](docs/reports/frequency_prototypes_audit_20260716.md)。

## 可复现实验

### FreqKV 历史结果重评

该实验只依赖 NumPy，不加载模型权重：

```bash
python -m unittest discover -s tests -v
python scripts/run_freqkv_reassessment.py --publish
```

它复现旧 Gaussian/RoPE-like 数据模型，对比正确 DCT、正确 rFFT、
历史丢虚部 rFFT，并使用显式平滑信号作为正对照。该结果是合成反证，
不是实际 LLM KV cache 的效果评测。

### FreqKV 真实 KV 频谱探针

该实验需要 PyTorch、Transformers、Datasets、SciPy、模型 checkpoint
和本地 WikiText-2 Arrow 数据：

```bash
pip install -e '.[model-eval]'
export CUDA_VISIBLE_DEVICES=0
python scripts/run_freqkv_real_kv_probe.py \
  --model-dir /path/to/Qwen2.5-1.5B \
  --test-arrow /path/to/wikitext-test.arrow \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 \
  --token-offsets 0,4096,8192,12288 \
  --sequence-length 512 \
  --output-dir runs/freqkv_real_kv \
  --publish
```

探针同时报告原始张量和逐头逐通道去序列均值后的 DCT/rFFT 统计，并
捕获 RoPE 前 K、RoPE 后 K 和 V cache。它不修改 attention，也不报告
压缩后模型质量或部署收益。

### 块级频域研究

块级实现只依赖 NumPy 和 safetensors，不使用 GPU：

```bash
export CUDA_VISIBLE_DEVICES=
python -m unittest discover -s tests -v
python scripts/run_transform_potential.py \
  --model-dir /data2/wangmeiqi/Llama-2-7b-chat-hf \
  --publish
```

正式运行使用 Llama-2-7B-chat 的第 0/8/16/24/31 层 Q 投影和 MLP
down-proj，每类每层 8 个 64×64 块，并用五个采样种子复核。聚合表见：

- [`docs/tables/transform_metrics_20260716.csv`](docs/tables/transform_metrics_20260716.csv)
- [`docs/tables/compression_rate_distortion_20260716.csv`](docs/tables/compression_rate_distortion_20260716.csv)
- [`docs/tables/transform_seed_sweep_20260716.csv`](docs/tables/transform_seed_sweep_20260716.csv)
- [`docs/tables/learned_rotation_20260716.csv`](docs/tables/learned_rotation_20260716.csv)
- [`docs/tables/kuramoto_phase_probe_20260716.csv`](docs/tables/kuramoto_phase_probe_20260716.csv)

这里测量的是变换域统计、重建误差和 layer-0 激活代理，不包含完整模型
旋转后的 perplexity、下游精度、真实 kernel 延迟或端到端存储收益。

### 模型级合法旋转

模型级实验需要 PyTorch、Transformers、Datasets、完整
Llama-2-7B base checkpoint 和本地 WikiText-2 Arrow 数据：

```bash
pip install -e '.[model-eval]'
python -m unittest discover -s tests -v
export FFT_COM_MODEL_DIR=/path/to/Llama-2-7b-hf
export FFT_COM_CALIBRATION_ARROW=/path/to/wikitext-train.arrow
export FFT_COM_TEST_ARROW=/path/to/wikitext-test.arrow
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_model_rotation_formal.sh
bash scripts/run_segment_replicates.sh
python scripts/summarize_model_rotation.py \
  --runs-dir runs \
  --latency-json runs/additional_8192/latency.json \
  --output-dir runs/final_summary \
  --publish
```

服务器路径、token 区段、量化规则、旋转位置和证据边界记录在
[`docs/protocols/legal_model_rotation_protocol.md`](docs/protocols/legal_model_rotation_protocol.md)。
发布的 JSON/CSV 不包含模型权重、原始数据或 calibration activation。

## 仓库结构

- `src/fft_com/`：DCT/Hadamard/RDFT、压缩、学习旋转、合法在线模型旋转与相位探针
- `scripts/`：块级实验、模型级 PPL、跨区段复核、延迟基准和聚合入口
- `tests/`：变换正交性、功能等价性、head/GQA 边界、量化和 Kuramoto sanity test
- `docs/reports/`：方法判断、证据审计与主线说明
- `docs/tables/`：可机器读取的结果表和方法对比表
- `docs/evidence/`：从服务器原始文件提取的轻量证据
- `docs/provenance/`：目录归属、上游提交号和文件哈希
- `docs/protocols/`：后续统一对比协议
- `comparisons/`：第三方对比项目的链接与本地快照边界

## 证据等级

- `rerun_test`：本次整理期间实际重新执行过
- `checkpoint_history`：从活动 checkpoint 历史直接读取，尚非最终结果
- `result_json`：原始小型结果文件仍存在
- `executed_notebook`：笔记本中保留了执行输出，但未形成独立指标文件
- `historical_executed_plot`：确认程序运行并生成图，但没有独立数值文件
- `source_document`：数值只存在于原项目说明文档
- `prototype_only`：只有代码、示意图或合成演示，不能支持效果结论
- `external_reference`：第三方已有工作，只记录来源

## 相关仓库

- 主线实现：<https://github.com/chaochao825/learnable_logic>，
  本次整理提交 `8aa65fbfe8650384584848c5236deac8fa717f57`
- DayPQ：<https://github.com/chaochao825/DayPQ>

## 不在本仓库发布的内容

- Llama/CIFAR 数据与模型权重
- 大型 checkpoint、优化器状态和训练缓存
- 第三方仓库源码副本
- 只有图片而没有数值协议的批量历史输出
- 尚未验证的“4× 压缩”“90%/85% 能量保持”等宣传性结论

## 可恢复清理

本次将 36 个明确的缓存、空目录、1 字节占位文件、调试输出、失败运行和
重复运行移入：

`/data2/wangmeiqi/trash/20260716-144759-fft-com-cleanup`

合计 1,950,853 bytes。没有永久删除；完整逐路径 manifest 留在该 trash
目录，分类摘要见
[`docs/provenance/cleanup_staging_20260716.tsv`](docs/provenance/cleanup_staging_20260716.tsv)。
