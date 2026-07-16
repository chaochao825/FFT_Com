# FFT_Com

这是服务器 34 上频域压缩、FFT 权重分析与离散逻辑主线的证据化整理仓库。
它不是旧目录的镜像，也不复制第三方仓库；目标是明确区分：

1. 已有工作的本地快照；
2. 自己做过的探索、失败尝试和可复用证据；
3. 当前值得继续投入的主线；
4. 后续进行公平对比所需的统一协议。

整理日期：2026-07-16。

## 当前结论

- **继续主线**：`learnable_logic` 的 full-discrete / ScaleLogic 交易级实现。
  其稳定价值是 Wmag7/A8、位平面、XNOR/popcount、确定性 Top-K、整数
  LUT 交易与部署导出，而不是目前尚未证明有效的 Hadamard 全局混合器。
- **频域工作保留为对比与重新立项素材**：`KV_FFT`、`Freq_KV`、
  `fourier_trans` 和 `order_test` 均不足以支持有效压缩或模型质量结论。
- **FFT 权重分析属于已运行的探索**：`test.ipynb` 曾读取本地
  Llama-2-7B 的 4096×4096 权重并生成行/列 FFT 图，但数值统计、二维
  FFT 分析与跨层比较没有执行，也没有独立结果表。
- **第三方项目只做来源记录**：`jacobfa/fft`、R-KV、ShadowKV 和
  monarch-attention 保持其上游身份，不纳入本仓库源码。

详细依据见
[`docs/reports/research_line_decision_20260716.md`](docs/reports/research_line_decision_20260716.md)
和
[`docs/reports/frequency_prototypes_audit_20260716.md`](docs/reports/frequency_prototypes_audit_20260716.md)。

## 仓库结构

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
