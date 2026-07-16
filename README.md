# FFT_Com

这是服务器 34 上频域压缩、正交旋转、FFT 权重分析与离散逻辑主线的
证据化整理仓库。它不是旧目录的镜像，也不复制第三方仓库；目标是明确
区分：

1. 已有工作的本地快照；
2. 自己做过的探索、失败尝试和可复用证据；
3. 当前值得继续投入的主线；
4. 后续进行公平对比所需的统一协议。

整理日期：2026-07-16。

## 当前结论

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
- **旧 FreqKV/fourier_trans/order_test 结论不变**：它们仍是失败原型
  或历史探索，不能支持压缩收益。

详细依据见
[`docs/reports/transform_potential_study_20260716.md`](docs/reports/transform_potential_study_20260716.md)、
[`docs/reports/research_line_decision_20260716.md`](docs/reports/research_line_decision_20260716.md)
和
[`docs/reports/frequency_prototypes_audit_20260716.md`](docs/reports/frequency_prototypes_audit_20260716.md)。

## 可复现实验

实现只依赖 NumPy 和 safetensors，不使用 GPU：

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

## 仓库结构

- `src/fft_com/`：DCT/Hadamard/FFT、压缩、学习旋转与相位探针
- `scripts/`：正式实验和多种子聚合入口
- `tests/`：变换正交性、Hermitian payload、量化和 Kuramoto sanity test
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
