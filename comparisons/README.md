# 第三方对比项目

这些目录是已有工作或外部上游快照。本仓库不复制其源码，也不主张其
方法或实现归属。

| 本地快照 | 上游 | 本地提交 | 状态/角色 |
|---|---|---|---|
| `FFTNet/fft` | <https://github.com/jacobfa/fft> | `d7eda7161cb56b0dd9ce7cf89adeefd53ca0e091` | 干净快照；SPECTRE/FFT 参考 |
| `/data2/wangmeiqi/fft` | <https://github.com/jacobfa/fft> | `5ff976ea7295e5b0dc0d261af9bd6a78a1d4c601` | 旧快照；含未跟踪的本地 `order_test/` |
| `KV_FFT/R-KV` | <https://github.com/Zefan-Cai/R-KV> | `d15c9767e67c2bd9807128fecc35223fa2484a69` | 干净快照；KV baseline |
| `KV_FFT/ShadowKV` | <https://github.com/ByteDance-Seed/ShadowKV> | `e51904cdeab7d4d34013370f09f2cf5fcd655e15` | `setup.py` 有本地修改；发布前不能视作原上游 |
| `monarch-attention` | <https://github.com/cjyaras/monarch-attention> | `cfe80d1679f313c01c2431bd03769e2a9313ec57` | 干净快照；结构化 attention 参考 |

使用这些项目进行实验时，应在结果 protocol 中记录精确 commit、补丁和
许可证，并把本地补丁单独保存。不能把第三方源码复制进 `FFT_Com`
后作为自己的实现发布。
