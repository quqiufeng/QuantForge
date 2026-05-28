# QuantForge

超低延迟 AI 量化交易系统 (OCaml 4 + RTX 3080 20G CUDA)

## 架构概述

完全抛弃 Python 运行时，实现亚毫秒级端到端延迟。

### 核心组件

| 组件 | 技术 | 用途 |
|------|------|------|
| 行情引擎 | OCaml 4 + Lwt | 高频 Tick 接入 |
| 推理核心 | LibTorch + CUDA | FP16 半精度推理 |
| 策略引擎 | Chez Scheme | Alpha 因子动态热更新 |

### RTX 3080 20G 优化

- **Pinned Memory**: cudaHostAlloc 固定内存，DMA 高速搬运
- **CUDA Streams**: non_blocking=true 异步传输
- **FP16 Tensor Core**: 推理算力翻倍，显存占用减半
- **零拷贝**: from_blob 直接包装 pinned memory

### 目录结构

```
quantforge/
├── Makefile                 # CUDA 编译配置
├── c_src/                   # C++/LibTorch CUDA 桥接层
│   ├── libquant_core.h      # 接口声明
│   └── libquant_core.cpp    # CUDA 实现
├── ocaml_src/               # OCaml 4 行情引擎
│   ├── bin/main.ml          # Lwt 异步主循环
│   └── lib/                 # 类型定义
└── scheme_src/              # Chez Scheme 策略
    ├── main.ss              # 策略主循环
    └── strategy.ss          # Alpha 因子
```

## 快速开始

```bash
# 验证 CUDA 环境
make verify-cuda

# 构建所有组件
make all

# 运行行情引擎
make run
```

## 路径配置

- LibTorch CUDA: `/data/libtorch_cuda`
- CUDA Toolkit: `/data/cuda`
- Chez Scheme: `/data/ChezScheme`

## License

See [LICENSE](LICENSE).
