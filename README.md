# QuantForge

**超低延迟 AI 量化交易系统 (OCaml 4 + RTX 3080 20G CUDA)**

![License](https://img.shields.io/badge/license-MIT-blue)
![OCaml](https://img.shields.io/badge/OCaml-4.14-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.6-green)
![LibTorch](https://img.shields.io/badge/LibTorch-2.3-red)

---

## 目录

- [项目概述](#项目概述)
- [设计哲学](#设计哲学)
- [系统架构](#系统架构)
- [技术方案详解](#技术方案详解)
- [核心模块说明](#核心模块说明)
- [性能优化](#性能优化)
- [安全机制](#安全机制)
- [快速开始](#快速开始)
- [路径配置](#路径配置)
- [构建指南](#构建指南)

---

## 项目概述

QuantForge 是一个跨语言混合的高性能量化交易系统，专为高频交易场景设计。系统完全抛弃 Python 运行时，从底层实现亚毫秒级端到端延迟。

### 核心特性

| 特性 | 实现方式 |
|------|----------|
| 亚毫秒延迟 | 单线程 Lwt 异步 + 无锁环形缓冲区 |
| 零拷贝通信 | Pinned Memory + Bigarray 裸指针映射 |
| GPU 加速 | CUDA Streams + FP16 Tensor Core |
| 动态策略 | Chez Scheme 运行时热更新 |
| 异常隔离 | C++ try-catch 边界保护 |

### 技术栈

| 层级 | 技术 | 版本 | 职责 |
|------|------|------|------|
| 行情接入 | OCaml 4 + Lwt | 4.14.1 | 高频 Tick 异步处理 |
| 推理核心 | LibTorch + CUDA | 2.3.0 + 12.6 | FP16 半精度推理 |
| 策略引擎 | Chez Scheme | 10.5 | Alpha 因子热更新 |
| 底层通信 | C/C++ | C++17 | 无锁缓冲区 + FFI 桥接 |

---

## 设计哲学

### 1. 拒绝 Python，拥抱原生

Python 的 GIL 锁、GC 停顿、解释器开销使其无法满足超低延迟需求。QuantForge 选择：

- **OCaml 4**：单线程 GC，类型安全，Lwt 异步零开销
- **Chez Scheme**：工业级 JIT，动态代码热更新
- **C++**：直接硬件控制，无运行时开销

### 2. 零拷贝，零分配

高频路径上的每一次内存分配都是延迟毛刺的来源：

```
传统方式：交易所 → JSON 解析 → 对象分配 → GC 管理 → 序列化 → 网络
QuantForge：交易所 → Pinned Buffer → GPU DMA → 直接推理
```

### 3. 无锁设计

避免互斥锁、条件变量等同步原语的开销：

```
OCaml 4 (写入)  ──► g_feature_buffer + atomic counter ──► Chez Scheme (轮询)
      │                                                        │
      └──── Release Barrier ──────────── Acquire Barrier ──────┘
```

### 4. 异常绝对隔离

C++ 异常绝不飞出 C ABI 边界，所有错误通过返回码传递：

```cpp
try {
    // LibTorch 推理
} catch (...) {
    return -999.0f;  // 安全降级
}
```

---

## 系统架构

### 整体拓扑

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           QuantForge 系统架构                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     WebSocket/UDP      ┌─────────────────────────┐
│   交易所网关     │ ◄───────────────────── │    行情源               │
│  (外部系统)      │                        │  (Binance/OKX/...)     │
└────────┬────────┘                        └─────────────────────────┘
         │
         │ Tick 数据流
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     OCaml 4 行情接入引擎                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Lwt 异步事件循环                                            │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐               │   │
│  │  │ WebSocket │  │   UDP     │  │  心跳监控  │               │   │
│  │  │  Handler  │  │ Multicast │  │           │               │   │
│  │  └─────┬─────┘  └─────┬─────┘  └───────────┘               │   │
│  │        │              │                                     │   │
│  │        └──────┬───────┘                                     │   │
│  │               │                                             │   │
│  │               ▼                                             │   │
│  │  ┌─────────────────────────────────────────┐               │   │
│  │  │  Bigarray.Array1 (裸指针映射)           │               │   │
│  │  │  ↓ Array1.set = 纯内存寻址 (~2ns)      │               │   │
│  │  └─────────────────────────────────────────┘               │   │
│  └─────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    │ 写入特征 + 递增序列号 (Release)
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  C++ 底层交互与推理桥接层                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Pinned Memory (cudaHostAlloc)                  │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │  g_feature_ring_buffer [1024 floats]                │   │   │
│  │  │  alignas(64) - 缓存行对齐                            │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  │                                                             │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                 │   │
│  │  │ g_sequence      │  │ g_heartbeat     │                 │   │
│  │  │ alignas(64)     │  │ alignas(64)     │                 │   │
│  │  └─────────────────┘  └─────────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                    │                               │
│                                    │ CUDA Stream 异步传输          │
│                                    │ non_blocking=true             │
│                                    ▼                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    RTX 3080 20G GPU                          │   │
│  │  ┌─────────────────┐  ┌─────────────────────────────────┐  │   │
│  │  │   VRAM          │  │  TorchScript 模型               │  │   │
│  │  │   (FP16)        │──│  FP16 Tensor Core 推理          │  │   │
│  │  │                 │  │  29.8 TFLOPS                    │  │   │
│  │  └─────────────────┘  └─────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    │ 推理结果
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Chez Scheme 策略引擎                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  高性能轮询循环                                               │   │
│  │  (polling g_sequence_counter - Acquire Barrier)             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                    │                               │
│                                    ▼                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Alpha 因子计算                                               │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐               │   │
│  │  │  动量因子  │  │ 均值回归  │  │  波动率   │               │   │
│  │  │           │  │   因子    │  │   因子    │               │   │
│  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘               │   │
│  │        │              │              │                      │   │
│  │        └──────┬───────┴──────┬───────┘                      │   │
│  │               │              │                              │   │
│  │               ▼              ▼                              │   │
│  │  ┌─────────────────────────────────────────┐               │   │
│  │  │     加权综合信号生成                      │               │   │
│  │  └─────────────────────────────────────────┘               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                    │                               │
│                                    ▼                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  交易指令输出                                                 │   │
│  │  • buy / sell / hold                                        │   │
│  │  • 价格、数量、时间戳                                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 数据流详解

```
时间线 ──────────────────────────────────────────────────────────────►

OCaml:    [接收Tick]→[写Buffer]→[序列号++]→[接收Tick]→...
                   ↓         ↓
C++:      ─────────[Release Barrier]────────────────────────────►
                                        ↓
Chez:     ─────────────────────────────[Acquire]→[读序列号]→[推理]→...
                                              ↓
CUDA:     ─────────────────────────────[异步传输]→[FP16推理]→[结果]→...
```

---

## 技术方案详解

### 1. 缓存行对齐与伪共享隔离

**问题**：多核 CPU 的缓存行 (64 字节) 竞争会导致严重的性能下降。

**场景**：
```
CPU Core 0 (OCaml)     CPU Core 1 (Chez Scheme)
        │                        │
        ▼                        ▼
   写 g_sequence           轮询 g_sequence
        │                        │
        └──── 同一缓存行 ────────┘
              ↓
   Core 0 写入 → Core 1 缓存行失效 → Cache Miss (~100ns)
```

**解决方案**：
```cpp
// 每个变量独占一个 64 字节缓存行
alignas(64) float g_feature_ring_buffer[1024];  // OCaml 写入区
alignas(64) uint64_t g_sequence_counter;         // 序列计数器
alignas(64) uint64_t g_ocaml_heartbeat;          // OCaml 心跳
alignas(64) uint64_t g_scheme_heartbeat;         // Scheme 心跳
```

**效果**：
```
CPU Core 0 (OCaml)     CPU Core 1 (Chez Scheme)
        │                        │
        ▼                        ▼
   缓存行 A                  缓存行 B
   (g_feature)              (g_sequence)
        │                        │
        └──── 独立缓存行 ────────┘
              ↓
   无竞争，轮询延迟 ~4ns
```

### 2. OCaml Bigarray 裸指针映射

**传统方式（慢）**：
```ocaml
let arr = Bigarray.Array1.create Float32 c_layout size
(* OCaml 堆分配 → GC 管理 → 边界检查 → ~50-100ns/元素 *)
```

**裸指针方式（快）**：
```ocaml
(* C 层预分配内存 *)
let c_ptr = C.Functions.get_feature_buffer_ptr ()

(* 直接映射 C 内存地址 *)
let arr = Ctypes.bigarray_of_ptr Ctypes.array1 size c_ptr

(* Array1.set 变成纯内存寻址 → ~1-2ns/元素 *)
Bigarray.Array1.set arr 0 value
```

**实现细节**：
```ocaml
(* c_ffi.ml - OCaml C FFI 绑定 *)
external get_feature_buffer_ptr : unit -> float Ctypes.ptr
  = "c_get_feature_buffer_ptr"

let feature_buffer = 
  Ctypes.bigarray_of_ptr Ctypes.array1 
    1024 
    (get_feature_buffer_ptr ())
```

### 3. Pinned Memory 与 CUDA 异步传输

**问题**：普通内存 → GPU 需要先复制到临时缓冲区，再 DMA 传输。

**Pinned Memory 优势**：
```cpp
// 普通内存分配
float* buffer = new float[1024];
// 传输路径: buffer → 临时 pinned buffer → GPU (2x 延迟)

// Pinned Memory 分配
cudaHostAlloc(&buffer, size, cudaHostAllocPortable);
// 传输路径: buffer → GPU DMA 直接访问 (1x 延迟)
```

**CUDA Stream 异步**：
```cpp
// 同步传输（阻塞）
tensor.to(torch::kCUDA);  // CPU 等待传输完成

// 异步传输（非阻塞）
tensor.to(torch::kCUDA, /*non_blocking=*/true);  // CPU 立即返回
// 传输在 CUDA Stream 后台进行，OCaml 可继续写入下一 Tick
```

**完整流程**：
```cpp
// 1. OCaml 写入 Pinned Memory
memcpy(g_feature_ring_buffer, tick_data, size);

// 2. 创建零拷贝 Tensor
auto tensor = torch::from_blob(g_feature_ring_buffer, shape);

// 3. 异步推送到 GPU
auto gpu_tensor = tensor.to(torch::kCUDA, torch::kHalf, true);
// ↑ CPU 立即返回，OCaml 可写入下一 Tick

// 4. 推理（内部自动同步）
auto result = model.forward(gpu_tensor);
```

### 4. FP16 半精度 Tensor Core 推理

**RTX 3080 Tensor Core 规格**：
| 精度 | 算力 | 显存占用 |
|------|------|----------|
| FP32 | 29.8 TFLOPS | 1x |
| FP16 | 59.6 TFLOPS | 0.5x |
| INT8 | 238 TOPS | 0.25x |

**FP16 转换**：
```cpp
// 加载模型后转换为 FP16
wrapper->module.to(torch::kHalf);

// 推理时一步完成 CPU→GPU + FP32→FP16
auto gpu_tensor = cpu_tensor.to(
    torch::Device(torch::kCUDA),
    torch::kHalf,
    /*non_blocking=*/true
);
```

**显存优化效果**：
```
模型大小: 500MB (FP32)
        → 250MB (FP16)
        
RTX 3080 20GB:
- FP32: 可加载 ~38 个模型
- FP16: 可加载 ~76 个模型
```

### 5. 内存快照克隆保护

**问题场景**：
```
时间 T1: LibTorch 开始 model->forward(input_tensor)
时间 T2: OCaml 收到新 Tick，覆写 g_feature_ring_buffer
时间 T3: 模型读取到被覆写的数据 → 推理结果错乱！
```

**解决方案**：
```cpp
torch::Tensor run_inference(void* model_ptr) {
    // 1. 零拷贝包装 C 缓冲区
    auto raw_view = torch::from_blob(g_feature_ring_buffer, shape);
    
    // 2. 克隆创建私有快照
    auto safe_snapshot = raw_view.clone();  // ~1-2μs
    
    // 3. 对快照执行推理
    // 此时即使 OCaml 覆写 buffer，推理数据不受影响
    return model.forward(safe_snapshot);
}
```

**性能权衡**：
```
clone() 开销: ~1-2μs (1024 floats = 4KB)
推理开销: ~100μs (典型模型)
占比: ~1-2% (可接受)
收益: 100% 数据一致性保证
```

### 6. 无锁同步与内存屏障

**写入端 (OCaml)**：
```cpp
void write_features(const float* data, size_t count) {
    // 1. 写入特征数据
    memcpy(g_feature_ring_buffer, data, count * sizeof(float));
    
    // 2. Release Barrier - 确保数据写入对后续 Acquire 可见
    atomic_thread_fence(memory_order_release);
    
    // 3. 递增序列号
    g_sequence_counter.fetch_add(1, memory_order_release);
}
```

**读取端 (Chez Scheme)**：
```scheme
(define (poll-and-process)
  (let ([seq (read-sequence)])  ; Acquire Barrier
    (when (> seq last-seq)
      ;; 安全读取特征数据
      (let ([result (run-inference)])
        (process-signal result)))))
```

**同步语义**：
```
Release (写入端):
  确保之前的所有写入操作，在 Release 点之前完成
  
Acquire (读取端):
  确ture 读取到 Release 之后的所有写入
```

---

## 核心模块说明

### 1. OCaml 4 行情接入引擎

**文件结构**：
```
ocaml_src/
├── bin/
│   └── main.ml          # 入口 + Lwt 主循环
└── lib/
    ├── types.ml         # 类型定义 (ADT)
    └── ingestion.ml     # 行情接入模块
```

**核心类型**：
```ocaml
(* 订单方向 *)
type side = Buy | Sell

(* 市场行情快照 *)
type tick = {
  symbol : string;
  bid : price;
  ask : price;
  bid_size : quantity;
  ask_size : quantity;
  timestamp : int64;   (* 微秒级 *)
  sequence : int64;
}

(* 订单簿 *)
type orderbook = {
  symbol : string;
  bids : price_level list;
  asks : price_level list;
  last_update : int64;
  sequence : int64;
}
```

**Lwt 异步循环**：
```ocaml
let rec ingestion_loop tick_source buffer count =
  let%lwt tick = tick_source () in
  (* 写入 C 缓冲区 *)
  write_tick_to_buffer buffer tick;
  (* 递归继续 *)
  ingestion_loop tick_source buffer (count + 1)
```

### 2. C++ 推理桥接层

**接口定义**：
```c
// 初始化 CUDA 和 Pinned Memory
int init_cuda_context(void);

// 加载 TorchScript 模型 (自动移至 GPU)
void* load_torch_model(const char* path);

// 执行推理 (异步传输 + FP16)
float run_inference(void* model_ptr);

// 获取 Pinned Memory 指针
float* get_feature_buffer_ptr(void);

// 无锁写入
void write_features(const float* data, size_t count);
uint64_t read_sequence(void);
```

### 3. Chez Scheme 策略引擎

**FFI 绑定**：
```scheme
;; 加载 C++ 共享库
(load-shared-object "./libquant_core.so")

;; 声明 FFI 接口
(define load-torch-model
  (foreign-procedure "load_torch_model" (string) void*))

(define run-inference
  (foreign-procedure "run_inference" (void*) float))

(define read-sequence
  (foreign-procedure "read_sequence" () unsigned-64))
```

**Alpha 因子定义**：
```scheme
(define-record-type alpha-factor
  (fields name weight calculate-fn enabled?))

(define momentum-factor
  (make-factor "momentum" 0.3
    (lambda (prices)
      (/ (- (car prices) (cadr prices))
         (cadr prices)))))
```

---

## 性能优化

### 延迟分解

| 阶段 | 操作 | 延迟 |
|------|------|------|
| 行情接入 | WebSocket 接收 | ~10μs |
| 解析 | 二进制/JSON 解析 | ~1μs |
| 写入 | Pinned Memory 拷贝 | ~100ns |
| 序列号 | atomic fetch_add | ~10ns |
| 轮询 | Chez Scheme 轮询 | ~4ns |
| 传输 | CUDA Stream 异步 | ~10μs |
| 推理 | FP16 Tensor Core | ~100μs |
| **总计** | | **~125μs** |

### 吞吐量

| 场景 | 吞吐量 |
|------|--------|
| Tick 写入 | ~10M/s (单核) |
| 序列号轮询 | ~250M/s |
| FP16 推理 | ~10K/s (单模型) |

---

## 安全机制

### 1. 心跳检测

```cpp
// OCaml 每 N 次 Tick 更新心跳
update_ocaml_heartbeat();

// Chez Scheme 每次策略循环更新心跳
update_scheme_heartbeat();

// 定期检查对方心跳
int health = check_system_health();
if (health != 0) {
    trigger_emergency_stop();
}
```

### 2. 断路器

```cpp
// 紧急停止标志
alignas(64) uint32_t g_emergency_stop;

// 推理前检查
float run_inference(void* model_ptr) {
    if (g_emergency_stop) {
        return -999.0f;  // 拒绝推理
    }
    // ... 正常推理流程
}
```

### 3. 异常隔离

```cpp
try {
    // LibTorch 推理
    auto output = model.forward(input);
    return output.item<float>();
} catch (const c10::Error& e) {
    std::cerr << "[ERROR] LibTorch: " << e.what() << std::endl;
    return -999.0f;  // 安全降级
} catch (...) {
    return -999.0f;  // 绝对不抛出异常
}
```

---

## 快速开始

### 环境要求

| 组件 | 版本 | 路径 |
|------|------|------|
| OCaml | 4.14+ | 系统 |
| CUDA | 12.6+ | `/data/cuda` |
| LibTorch | 2.3+ (CUDA) | `/data/libtorch_cuda` |
| Chez Scheme | 10.5+ | `/data/ChezScheme` |

### 构建

```bash
# 克隆仓库
git clone https://github.com/quqiufeng/QuantForge.git
cd QuantForge

# 安装依赖
make deps

# 验证 CUDA 环境
make verify-cuda

# 构建全部组件
make all
```

### 运行

```bash
# 运行行情引擎
make run

# 运行策略引擎
make run-scheme
```

---

## 路径配置

```makefile
# Makefile 中的路径配置
LIBTORCH_DIR = /data/libtorch_cuda
CUDA_DIR = /data/cuda
SCHEME_BIN = /data/ChezScheme/pb/bin/pb/petite
```

---

## 构建指南

### Makefile 目标

| 目标 | 说明 |
|------|------|
| `all` | 构建所有组件 |
| `ocaml` | 构建 OCaml 引擎 |
| `c_lib` | 构建 C++ CUDA 库 |
| `scheme` | 准备 Chez Scheme 模块 |
| `test` | 运行测试 |
| `clean` | 清理构建产物 |
| `run` | 运行行情引擎 |
| `verify-cuda` | 验证 CUDA 环境 |

---

## License

[MIT License](LICENSE)

---

## 致谢

- [OCaml](https://ocaml.org/) - 工业级函数式语言
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Chez Scheme](https://cisco.github.io/ChezScheme/) - 高性能 Scheme 实现
- [Lwt](https://ocsigen.org/lwt/) - OCaml 协程库

## 设计文档

| 文档 | 说明 |
|------|------|
| [设计规格说明书](docs/DESIGN_SPEC.md) | 完整的五章架构设计 |
| [实现规格补充修正案](docs/DESIGN_SPEC_ADDENDUM.md) | ABI稳定性、Tensor解包、GC协同、内存屏障 |
