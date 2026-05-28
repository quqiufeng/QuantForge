# QuantForge

超低延迟 AI 量化交易系统 (OCaml 4 稳定版)

## 架构概述

QuantForge 是一个跨语言混合的高性能量化交易系统，完全抛弃 Python 运行时，实现亚毫秒级端到端延迟。

### 核心组件

- **OCaml 4** (行情接入与风控): 单核单线程架构，基于 Lwt 异步事件循环
- **Chez Scheme** (策略引擎与动态沙箱): 支持 Alpha 因子实时热更新
- **LibTorch C++** (深度学习核心): 纯 C++ 环境下的高性能前向推理

### 目录结构

```
quantforge/
├── Makefile                 # 统一构建脚本
├── dune-project             # OCaml 项目定义
├── c_src/                   # C++ / LibTorch 桥接层
├── ocaml_src/               # OCaml 4 行情接入引擎
│   ├── bin/main.ml          # Lwt 异步主循环入口
│   └── lib/                 # 类型定义与接入模块
└── scheme_src/              # Chez Scheme 策略沙箱
```

## 快速开始

### 安装依赖

```bash
make deps
```

### 构建项目

```bash
make all
```

### 运行行情接入引擎

```bash
make run
```

## 设计原则

1. **零拷贝 (Zero-Copy)**: 数据交互基于无锁环形缓冲区
2. **单线程 GC 隔离**: OCaml 4 单线程运行时避免 GC 抖动
3. **异常绝对隔离**: C++ 异常不跨越 ABI 边界
4. **无锁通知**: 原子计数器实现进程间同步

## License

See [LICENSE](LICENSE) file.
