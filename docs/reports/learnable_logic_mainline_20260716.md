# Learnable Logic 主线摘要

日期：2026-07-16
发布仓库：<https://github.com/chaochao825/learnable_logic>
整理提交：`8aa65fbfe8650384584848c5236deac8fa717f57`

## 方法价值

当前最稳定的研究对象是 full-discrete carrier：

- 激活为带二次幂 exponent 的 signed A8 code；
- 权重为 sign + 7 个 magnitude planes，范围 `[-127,127]`；
- A8×U4 使用 4096-entry 精确产品 LUT，高三位按 `<<4` 重构；
- Q/K 阈值化后通过 packed XNOR/popcount 计算相似度；
- hard Top-K 的 tie-break 固定为 score 降序、key index 升序；
- 选中 value 使用 `{8,4,2,1}` score-gap 权重；
- 支持 Q0.15 RMS-LUT、Shift-RMS、requant 和 no-norm 对照；
- 导出路径移除训练态 FP32 shadow、优化器和 surrogate-only 对象。

这是交易级 Python 参考实现，不是完成的 RTL。尚缺端到端整数
accumulator/residual→A8 requant、位宽冻结、高吞吐 kernel、综合和能耗
测量。

## 已有 50k 结果

这些数值来自源 README/探针文档；服务器快照中没有全部原始结果文件，
因此证据等级为 `source_document`。

- Wmag7 + Q15 RMS：75.30%
- Shift-RMS block + final：71.80%
- Shift-RMS block + no final norm：74.60%
- RMS block + no final norm：74.74%
- requant-only：70.10%
- no norm：70.64%
- logic-tree local 0/1/3：75.30% / 75.76% / 58.28%

logic-tree 实验中 hard LUT 全部保持 `0xC`，hard root change rate 为 0。
单层 `+0.46` 个百分点不能归因于部署后的新布尔表达式。

## Mixer smoke

1k-step d6/e192：

- attention local-6：49.30%
- LHVM parallel：47.42%
- LHVM hybrid：31.92%
- LHVM：26.66%

这些只证明代码路径可运行，不支持最终方法优劣。当前没有证据表明
Hadamard/LHVM 优于 attention。

## 活动 ScaleLogic 50k

配置：

- CIFAR-10，seed 42，计划 50k steps
- d12/e384/h12，Top-K 8，Wmag7/A8，7 条 Q/K lanes
- Q15 RMS-LUT block/final norm
- global mixer：attention
- 前四层 local operator：depthwise shift-add

截至本次快照：

| Step | Validation accuracy | Train loss | Peak GiB | 秒/step |
|---:|---:|---:|---:|---:|
| 5,000 | 0.5958 | 1.4691 | 11.7573 | 0.4120 |
| 10,000 | 0.6664 | 1.1960 | 11.7559 | 0.5500 |
| 15,000 | 0.6830 | 1.0756 | 11.7559 | 0.7802 |
| 20,000 | 0.6960 | 1.0678 | 11.7559 | 0.4145 |

运行仍在继续，不能将 69.60% 写成最终准确率。

## 判断

继续 full-discrete/ScaleLogic；Hadamard 和频域方法保持为统一协议下的
候选对比。下一轮优先级是：

1. 完成冻结协议的 paired control；
2. 多种子；
3. 闭合整数 requant 边界；
4. 再做 mixer 与频域基线比较；
5. 最后进入 kernel/RTL/PPA。
