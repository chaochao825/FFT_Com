# 频域原型证据审计

日期：2026-07-16

## 1. FFTNet 根目录：权重频谱探索

### 实际存在的证据

`test.ipynb` 曾在 `awq` 环境中成功加载本地
`Llama-2-7b-chat-hf`，提取一个 4096×4096 的权重矩阵，并完成逐行、
逐列 FFT 图。笔记本保留了 8 个已执行代码单元和 8 个 PNG 输出。

但以下单元没有执行：

- 二维 FFT 可视化；
- 数值统计；
- 不同层/不同权重类型比较。

因此它证明“代码曾对真实模型权重生成过图”，不证明频域可压缩性，也
没有压缩后质量、误差或性能指标。

另外：

- `fft_analysis_for_llm_weights.ipynb` 只有一个未执行代码单元，仍使用
  `/path/to/llama-2-7b` 占位路径；
- `advanced_fft_analysis.ipynb` 只有 1 字节空格；
- `fft_utils.py` 可以通过语法编译，但当前基础环境因 NumPy 2 与
  SciPy/Pandas/PyArrow 等二进制 ABI 冲突而无法导入；
- 没有锁定环境、随机种子、独立 CSV/JSON 或跨模型结果。

判断：保留为“真实权重频谱探索”的历史证据；若继续，应重新建立可复现
环境并输出数值表，而不是继续依赖嵌入式图片。

## 2. KV_FFT / FreqKV

### 变换和环境问题

- 当前 PyTorch 没有脚本尝试调用的 `torch.fft.dct/idct`。
- SciPy DCT 路径在现有基础环境中被 NumPy ABI 冲突阻断。
- fallback 使用 `rfft`，但逆变换前把 complex 系数强制转成 float，
  丢弃虚部。
- `KV` conda 环境缺少 torch/scipy/pandas/seaborn；基础环境虽能找到
  包，却不能稳定导入。

### 合成数据不能支持能量集中结论

示例 K/V 从独立高斯噪声开始，再逐位置施加正交 RoPE-like 旋转。
该操作不会把白噪声变成低频信号。用正交 DCT 对 5 个种子独立复核，
前 25%/50%/75% 系数的能量占比约等于保留比例：

| Seed | 前 25% | 前 50% | 前 75% |
|---:|---:|---:|---:|
| 0 | 0.24995 | 0.49891 | 0.75637 |
| 1 | 0.25251 | 0.49793 | 0.74106 |
| 2 | 0.26274 | 0.50867 | 0.76002 |
| 3 | 0.24952 | 0.50068 | 0.74874 |
| 4 | 0.23537 | 0.48191 | 0.73823 |

原脚本中的 `0.90` 和 `0.85` 是 `simulate_freqkv_compression` 内的硬编码
占位值，并非测量结果。README 中的“90%+ K / 85%+ V”不能作为证据。

### Cache 语义和内存问题

- 增量更新会重复追加当前 token。简单探针得到：
  `[0,1,1] → [0,1,1,2,2] → [0,1,1,2,2,3,3]`。
- 压缩函数把序列长度从 N 变为 L，但能量分析直接将原始和压缩张量
  相减，出现 shape mismatch。
- `BlockCirculantFFT` 数值上只是普通正交 DFT，同时构造一个未使用的
  `S³` float32 basis；S=4096 时理论占用 256 GiB。
- `compress_kv_cache_fft` 返回原始 K/V、完整 FFT、全尺寸零填充张量和
  重建张量，没有形成 compact payload，也不能证明实际显存下降。
- `freqkv_model.py` 与现代 Hugging Face cache/GQA 接口没有验证；RoPE
  异常会被静默吞掉。

### 缺失的关键证据

没有找到：

- perplexity、LongBench、RULER、NIAH 或 lm-eval 结果；
- 真实模型 KV capture；
- baseline 与压缩模型的同协议质量对比；
- 峰值显存、常驻 KV 字节数或延迟数据；
- 可复现 environment lock；
- compact serialization 或实际部署后端。

判断：这是未完成的合成原型。可以保留设计问题与重启要求，但不应继续
传播其效果结论。

## 3. fourier_trans

`compute_fft_dot_product` 计算完整循环相关后对所有 lag 求和。正确点积
应取 lag 0。独立复核的最大误差：

| 长度 | 当前实现最大误差 | lag-0 公式误差 |
|---:|---:|---:|
| 8 | 3.6878 | 9.5e-7 |
| 16 | 23.5623 | 4.8e-7 |
| 64 | 42.9224 | 1.4e-6 |

此外：

- 频域“卷积”路径求幅值和，不是点积；
- `run_analysis.py` 使用未定义的 `args.max_memory`；
- `demo_2.py` 有多个未定义变量；
- `demo_3.py` 在定义前使用 `sparse_results`；
- 没有可复核 JSON 结果。

判断：作为失败的数学探索归档；原实现停止继续。

## 4. order_test

该目录是第三方 `jacobfa/fft` 仓库中的本地未跟踪增量。

- 输入是合成高斯权重/激活，不是模型张量；
- “shuffle”分别抽取 source 和 destination，并发生覆盖/碰撞，不是
  真正的子集置换；
- 只有 weight test 完成，attention 测试日志在启动后停止；
- 两次完成结果字节完全一致，SHA256 都是
  `3b0465c6305da273a108b6ea0c7ef17975bed3ed67d88ef26f232be275a9a0e0`；
- 余弦相似度近似 `1 - shuffle_fraction`，属于可预期的随机线性代数
  现象，不是频域压缩证据。

判断：保留一个 JSON 作为历史证据；失败运行和字节重复运行移入 trash。

## 5. 如果重新启动频域方向

最低可接受起点：

1. 从真实模型、真实任务和真实 KV capture 开始；
2. 明确变换轴、RoPE 前后位置、sink/recent token 策略；
3. 保存真正紧凑的系数和索引，不保留全尺寸零填充副本；
4. 对 cache 长度、位置和 token 追加写性质测试；
5. 同时报告质量、常驻字节、临时 buffer、延迟和失败；
6. 与原 attention、R-KV、ShadowKV 以及 full-discrete 主线使用同一协议。
