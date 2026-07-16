# 统一比较协议

任何 frequency-KV、attention/Hadamard mixer 和 full-discrete 方法，只有
满足本协议后才进入同一结论表。

Hadamard、DCT、FFT、learned rotation 和 Kuramoto-inspired 权重压缩还
必须满足
[`transform_compression_protocol.md`](transform_compression_protocol.md)。

## 冻结项

- 模型结构、参数量、tokenizer、数据版本和 split
- seed 列表、训练/校准 steps、batch size、优化器和学习率计划
- context length、prefill/decode 设置和 generation 参数
- baseline commit、方法 commit、环境 lock 和启动命令
- 失败、OOM、重试和手工中断记录

## 质量指标

- 训练任务：validation accuracy/loss，多种子 mean/std
- 语言模型：perplexity 和至少一个长上下文任务
- KV 方法：按位置的 retrieval/QA 指标和 context sweep
- 报告 hard/deployable 路径，不只报告 soft/fake-quant 路径

## 存储指标

必须分别记录：

- 参数常驻字节
- KV payload 常驻字节
- scale、index、mask、sink/recent token 元数据
- FFT/DCT 临时 buffer
- 全尺寸零填充或重建 buffer
- 峰值 allocated/reserved memory

“压缩率”使用实际存储 payload 字节计算，不使用非零元素比例代替。

## 计算指标

- MAC/add/shift/LUT/popcount/FFT 操作计数
- prefill 与 decode 延迟、tokens/s
- 相同硬件、batch、context 和 warm-up
- 若进入硬件结论，报告综合条件、频率、面积、功耗和 memory traffic

## Frequency-KV 必要测试

- append 一个 token 后长度精确增加 1
- 位置 id 与 cache token 一一对应
- RoPE 前/后变换约定明确
- compact payload 可序列化并独立解码
- reconstruction shape 与原始 shape 一致
- Parseval/能量与误差计算维度正确
- 白噪声 sanity check 不应产生虚假低频集中

## 证据发布

每次运行至少保存：

- `protocol.json`
- `metrics.jsonl`
- `result.json`
- source/config SHA256
- 环境摘要
- 失败状态或 exit code
