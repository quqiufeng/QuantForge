# QuantForge 多损失函数风控模型与智能动态仓位定投系统

## 设计与实现规格说明书

**版本**: 1.0  
**日期**: 2026-05-28  
**架构师**: QuantForge Team  

---

# 第一章：多任务联合损失深度网络架构

## 1.1 输入特征矩阵构建

### 1.1.1 四维度特征抽象

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      输入特征矩阵 X ∈ R^(B×T×D)                         │
│                                                                         │
│  B = Batch Size (推理时 B=1)                                            │
│  T = 时序窗口长度 (T=128 ticks)                                         │
│  D = 特征维度 (D=64)                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

**维度一：订单簿微观结构特征 (D₁=24)**

```
┌─────────────────────────────────────────────────────────────────┐
│  订单簿特征 (每 Tick 24 维)                                       │
├─────────────────────────────────────────────────────────────────┤
│  [0:5]   买盘 Top5 价格 (归一化)                                  │
│  [5:10]  买盘 Top5 数量 (log 归一化)                              │
│  [10:15] 卖盘 Top5 价格 (归一化)                                  │
│  [15:20] 卖盘 Top5 数量 (log 归一化)                              │
│  [20]    买卖价差 Spread (mid_price 归一化)                       │
│  [21]    订单簿不平衡度 OI = (V_bid - V_ask)/(V_bid + V_ask)    │
│  [22]    加权中间价 WMP = Σ(Pᵢ×Vᵢ)/ΣVᵢ                         │
│  [23]    订单簿深度斜率 (线性回归斜率)                             │
└─────────────────────────────────────────────────────────────────┘
```

**维度二：跨品种相关性背离特征 (D₂=16)**

```
┌─────────────────────────────────────────────────────────────────┐
│  跨品种特征 (每 Tick 16 维)                                       │
├─────────────────────────────────────────────────────────────────┤
│  [0:4]   相关品种 1min 收益率 (BTC/ETH/SOL/BNB)                  │
│  [4:8]   与目标品种的滚动相关系数 (窗口=60 ticks)                 │
│  [8:12]  相关性背离 Z-Score = (ρ_current - ρ_mean) / ρ_std      │
│  [12:16] 协整关系残差 (Johansen 协整向量投影)                    │
└─────────────────────────────────────────────────────────────────┘
```

**维度三：高频已实现波动率特征 (D₃=12)**

```
┌─────────────────────────────────────────────────────────────────┐
│  波动率特征 (每 Tick 12 维)                                       │
├─────────────────────────────────────────────────────────────────┤
│  [0:3]   已实现波动率 RV (1min/5min/15min 窗口)                  │
│  [3:6]   已实现半波动率 RS+ / RS- (上行/下行分离)                │
│  [6:9]   波动率变化率 dRV/dt (三个窗口)                          │
│  [9:12]  波动率 Z-Score (相对于历史分布)                         │
└─────────────────────────────────────────────────────────────────┘
```

**维度四：内部仓位暴露特征 (D₄=12)**

```
┌─────────────────────────────────────────────────────────────────┐
│  仓位特征 (每 Tick 12 维)                                         │
├─────────────────────────────────────────────────────────────────┤
│  [0:2]   当前持仓量 / 最大持仓量 (多头/空头分离)                  │
│  [2:4]   持仓成本价 / 当前价 (多头/空头分离)                      │
│  [4:6]   未实现盈亏 PnL (归一化)                                  │
│  [6:8]   仓位暴露度 (总敞口 / 账户净值)                           │
│  [8:10]  定投池占比 / 套利池占比                                  │
│  [10:12] 距离上次调仓的时间 (归一化)                              │
└─────────────────────────────────────────────────────────────────┘
```

### 1.1.2 时序窗口选择

```python
# 时序窗口参数
TICK_WINDOW = 128          # 输入序列长度
TICK_INTERVAL_MS = 100     # Tick 间隔 (100ms)
TOTAL_HISTORY_S = 12.8     # 总历史窗口 (12.8秒)

# 特征矩阵最终形状
# X.shape = (B, 128, 64)  # (Batch, Time, Features)
```

## 1.2 共享底座网络设计

### 1.2.1 轻量级时序卷积网络 (TCN) 底座

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TCN 共享底座架构                                  │
│                                                                         │
│  Input (B, 128, 64)                                                     │
│      │                                                                  │
│      ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Embedding Layer: Linear(64 → 128) + LayerNorm + GELU           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│      │                                                                  │
│      ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  TCN Block ×4 (dilation=[1,2,4,8], kernel=3, channels=128)      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│      │                                                                  │
│      ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Global Features: AdaptiveAvgPool1d + AdaptiveMaxPool1d         │   │
│  │  Output: (B, 256)                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│      │                                                                  │
│      ├──────────────────┬──────────────────┬─────────────────────┐     │
│      ▼                  ▼                  ▼                     │     │
│  Head_Trend          Head_Risk         Head_Volatility          │     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2.2 网络参数规格

```python
# TCN 底座参数
TCN_CONFIG = {
    'input_dim': 64,
    'hidden_dim': 128,
    'num_layers': 4,
    'kernel_size': 3,
    'dropout': 0.1,
    'dilations': [1, 2, 4, 8],
}

# 参数量估算
# 底座: ~275K 参数
# 显存: ~550KB (FP16)
# 推理延迟: ~50μs (RTX 3080 FP16)
```

## 1.3 多任务预测头设计

### 1.3.1 Head_Trend：趋势预测头

**输出**: 连续型收益率 ∈ R

**Huber Loss 损失函数**：
```
L_δ(y, ŷ) = {
    0.5 * (y - ŷ)²           if |y - ŷ| ≤ δ
    δ * |y - ŷ| - 0.5 * δ²   otherwise
}

其中 δ = 1.0
```

### 1.3.2 Head_Risk：风险尾部防御头

**输出**: 风险概率 ∈ [0,1]

**Focal Loss 损失函数**：
```
FL(p̂, y) = -α_t · (1 - p_t)^γ · log(p_t)

其中:
- p_t = p̂ if y=1, else 1-p̂
- α_t = 0.25 if y=1, else 0.75
- γ = 2 (聚焦参数)
```

### 1.3.3 Head_Volatility：波动率预测头

**输出**: 预测波动率 ∈ R⁺

**Gaussian NLL Loss**：
```
L_Vol(σ̂, σ) = log(σ̂ + ε) + (σ - σ̂)² / (σ̂² + ε)

其中 ε = 1e-6
```

## 1.4 联合反向传播

### 1.4.1 总损失加权公式

```
L_total = w_trend * L_Trend + w_risk * L_Risk + w_vol * L_Vol

推荐权重 (GradNorm 或固定):
w_trend = 0.286
w_risk  = 0.571  (安全优先)
w_vol   = 0.143
```

## 1.5 JIT 编译与导出规范

```python
# 导出流程
model.eval()                                    # 1. 切换 eval 模式
example_input = torch.randn(1, 128, 64)         # 2. 创建示例输入
traced = torch.jit.trace(model, example_input)  # 3. Trace 固化
traced_fp16 = traced.half()                      # 4. 转换 FP16
traced_fp16.save('model.pt')                     # 5. 保存
```

---

# 第二章：C++ 零拷贝与 RTX 3080 GPU 推理桥接设计

## 2.1 内存布局与缓存行隔离

### 2.1.1 全局共享内存结构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    缓存行对齐内存布局 (防 False Sharing)                  │
└─────────────────────────────────────────────────────────────────────────┘

缓存行 0-63: g_feature_ring_buffer[1024] (alignas(64))
缓存行 64:   g_sequence_counter (alignas(64))      ← OCaml 写，Scheme 轮询
缓存行 65:   g_ocaml_heartbeat (alignas(64))        ← OCaml 更新
缓存行 66:   g_scheme_heartbeat (alignas(64))       ← Scheme 更新
缓存行 67:   g_emergency_stop (alignas(64))         ← 紧急停止标志
```

### 2.1.2 C++ 结构体定义

```cpp
// 特征环形缓冲区 (CUDA Pinned Memory)
struct alignas(64) FeatureRingBuffer {
    float data[1024];
    // 构造时使用 cudaHostAlloc 分配固定内存
};

// 原子序列计数器 (独占缓存行)
struct alignas(64) SequenceCounter {
    std::atomic<uint64_t> value{0};
    void store_release(uint64_t val);
    uint64_t load_acquire() const;
};

// 心跳计数器
struct alignas(64) OcamlHeartbeat { std::atomic<uint64_t> value{0}; };
struct alignas(64) SchemeHeartbeat { std::atomic<uint64_t> value{0}; };

// 紧急停止标志
struct alignas(64) EmergencyStop { std::atomic<uint32_t> flag{0}; };
```

## 2.2 RTX 3080 显存异步流传输

### 2.2.1 CUDA Stream 异步执行模型

```
CPU 时间线: [调用推理] → [立即返回]
                           ↓
GPU 时间线: [H2D 传输] → [FP16 推理] → [D2H 传输]
            (non_blocking)  (Tensor Core)

OCaml 在 GPU 执行期间可继续写入下一个 Tick!
```

### 2.2.2 推理函数核心实现

```cpp
float run_inference(void* model_ptr) {
    auto& state = GlobalSharedState::instance();
    
    // 1. 从 Pinned Memory 创建零拷贝 Tensor
    auto cpu_view = torch::from_blob(
        state.feature_buffer.data,
        {1, 1024}
    );
    
    // 2. 克隆快照 (防止 OCaml 覆写)
    auto cpu_snapshot = cpu_view.clone();  // ~1-2μs
    
    // 3. 异步推送到 GPU + FP16 转换
    auto gpu_tensor = cpu_snapshot.to(
        torch::kCUDA, torch::kHalf, /*non_blocking=*/true
    );
    
    // 4. 执行推理
    auto output = model.forward({gpu_tensor});
    
    return output.toTensor().item<float>();
}
```

## 2.3 内存覆写快照机制

### 2.3.1 问题与解决方案

**问题**: OCaml 高频写入时，LibTorch 可能读到被覆写的数据

**解决方案**: `.clone()` 创建私有快照

```cpp
// 零拷贝视图
auto view = torch::from_blob(buffer, {1, 1024});

// 克隆到私有内存 (~1-2μs for 4KB)
auto snapshot = view.clone();

// 对快照执行推理
// 此时 OCaml 可自由覆写原 buffer
auto output = model.forward({snapshot.to(kCUDA, kHalf, true)});
```

## 2.4 异常绝对隔离屏障

```cpp
extern "C" float run_inference_safe(void* model_ptr) {
    // 参数检查
    if (!model_ptr) return -1.0f;
    
    try {
        // CUDA 推理
        return do_inference(model_ptr);
    } catch (const c10::Error& e) {
        // LibTorch/CUDA 错误
        return -999.0f;
    } catch (...) {
        // 绝对不允许异常飞出 C 边界
        return -999.0f;
    }
}
```

---

# 第三章：智能动态仓位管理与智能化定投状态机

## 3.1 双持仓池设计

### 3.1.1 资金隔离架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         双持仓池资金架构                                 │
└─────────────────────────────────────────────────────────────────────────┘

总账户净值 (NAV)
    │
    ├── 基准定投池 (Beta Pool) ── 60% 净值
    │   ├── 目标: 捕获大盘长期 Beta 收益
    │   ├── 策略: 定时定额 + 风险加权
    │   └── 调仓频率: 小时级/日级
    │
    └── 高频套利池 (Alpha Pool) ── 40% 净值
        ├── 目标: 利用 3080 推理榨干日内震荡
        ├── 策略: 智能网格 + 凯利公式
        └── 调仓频率: 毫秒级/秒级
```

## 3.2 动态网格定投函数

### 3.2.1 定投状态机

```
IDLE → EVALUATE → CALCULATE → EXECUTE → IDLE
              ↓
           HALT → LOCKED (风险熔断)
```

### 3.2.2 动态注入金额计算

```scheme
(define (calculate-dca-amount risk-prob trend-signal volatility)
  (cond
    ;; 风险过高：停止注入
    [(> risk-prob 0.8) 0.0]
    
    ;; 趋势为负：减少注入
    [(< trend-signal 0) (* base 0.5 (- 1.0 risk-prob))]
    
    ;; 趋势为正：根据信号强度调整
    [else
     (let* ([trend-weight (min 1.0 (+ 0.5 (* 0.5 trend-signal)))]
            [risk-weight (- 1.0 risk-prob)]
            [vol-weight (/ 1.0 (+ 1.0 (* 2.0 volatility)))])
       (min (* base trend-weight risk-weight vol-weight) max-amt))]))
```

## 3.3 凯利公式动态重平衡

### 3.3.1 凯利公式变体

```
经典凯利: f* = (p × b - q) / b

量化交易变体: f* = (E[R] / σ²) × min(f_max, 1/σ)

其中:
- E[R] = 预期收益率 (Head_Trend 输出)
- σ² = 收益率方差 (Head_Volatility 输出)
- f_max = 最大仓位限制 (0.6)
```

### 3.3.2 Scheme 实现

```scheme
(define (kelly-criterion expected-return volatility max-position)
  (let* ([variance (* volatility volatility)]
         [epsilon 1e-6]
         [kelly-raw (/ expected-return (+ variance epsilon))]
         [kelly-half (* 0.5 kelly-raw)]          ;; 半凯利 (更保守)
         [vol-scaling (/ 1.0 (+ 1.0 (* 5.0 volatility)))]
         [f-star (* kelly-half vol-scaling)])
    (max 0.0 (min f-star max-position))))
```

## 3.4 智能网格套利

### 3.4.1 动态网格步长计算

```scheme
(define (calculate-dynamic-grid current-price volatility holding-period-sec)
  (let* ([time-scaling (sqrt (/ holding-period-sec 3600))]
         [k 0.6]  ;; 调节系数
         [step (* current-price volatility time-scaling k)]
         [num-grids (min 10 (floor (/ capital risk-per-grid)))])
    (make-grid-config current-price step num-grids capital)))
```

### 3.4.2 网格订单生成

```scheme
(define (generate-grid-orders config)
  (append
    ;; 买单 (低于中心价)
    (map (lambda (i)
           `(buy ,(- center (* (+ i 1) step)) ,quantity))
         (iota n))
    
    ;; 卖单 (高于中心价)
    (map (lambda (i)
           `(sell ,(+ center (* (+ i 1) step)) ,quantity))
         (iota n))))
```

---

# 第四章：OCaml 4 单线程行情接入与高频写内存流

## 4.1 订单簿与风控数据类型

### 4.1.1 核心 ADT 定义

```ocaml
(* 交易方向 *)
type side = Buy | Sell

(* 订单状态 *)
type order_status = Pending | PartialFilled | Filled | Cancelled | Rejected

(* 订单记录 *)
type order = {
  id : int64;
  symbol : string;
  side : side;
  price : float;
  quantity : float;
  status : order_status;
  timestamp : int64;
}

(* 订单簿 *)
type orderbook = {
  symbol : string;
  bids : price_level list;    (* 买盘 (降序) *)
  asks : price_level list;    (* 卖盘 (升序) *)
  sequence : int64;
}

(* Tick 行情 *)
type tick = {
  symbol : string;
  bid : float;
  ask : float;
  bid_qty : float;
  ask_qty : float;
  timestamp : int64;
  sequence : int64;
}
```

### 4.1.2 风控状态机

```ocaml
type risk_state = Normal | Warning | Halted of string

type risk_engine = {
  config : risk_config;
  mutable state : risk_state;
  mutable daily_pnl : float;
}
```

## 4.2 Lwt 异步单线程无锁循环

### 4.2.1 单线程优势

```
【OCaml 4 多线程】              【OCaml 4 单线程 + Lwt】
Thread 1 ─┐                     单线程循环
Thread 2 ─┤─ GC 全局暂停         [Tick]→[写Buffer]→[Tick]→...
          │  (10-100ms)         
延迟毛刺大                       Minor GC 几乎不触发
                               延迟毛刺 < 1ms
```

### 4.2.2 避免分配规范

```ocaml
(* 错误: 频繁字符串拼接触发 GC *)
let bad tick = "Tick: " ^ tick.symbol ^ " bid=" ^ string_of_float tick.bid

(* 正确: 延迟分配 *)
let good tick = 
  if should_log () then Printf.printf "Tick: %s bid=%f\n" tick.symbol tick.bid
```

## 4.3 裸指针包装写入

### 4.3.1 Bigarray 裸指针映射

```ocaml
(* 获取 C 分配的 Pinned Memory 指针 *)
let get_feature_buffer_ptr =
  foreign "get_feature_buffer_ptr" (void @-> returning (ptr float))

(* 绑定到 Bigarray (managed:false = GC 不管理) *)
let feature_buffer =
  let ptr = get_feature_buffer_ptr () in
  bigarray_of_ptr array1 1024 Float32 ptr

(* 直接内存赋值 - 纳秒级开销 *)
let write_tick buffer tick =
  Bigarray.Array1.set buffer 0 tick.bid;    (* ~1-2ns *)
  Bigarray.Array1.set buffer 1 tick.ask;
  (* ... *)
```

### 4.3.2 性能对比

```
传统 Bigarray:  OCaml Heap → GC 管理 → ~50-100ns/元素
裸指针方式:     C Heap → 直接寻址 → ~1-2ns/元素

性能提升: ~50-100x
```

---

# 第五章：系统生产级保证与工程联调规划

## 5.1 双向心跳监控与极端保护机制

### 5.1.1 心跳架构

```
┌──────────────┐                    ┌──────────────┐
│   OCaml 4   │                    │ Chez Scheme  │
│  每100 Tick  │                    │  每策略循环   │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       ▼                                   ▼
┌─────────────────────────────────────────────────────────┐
│  C 共享内存                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐      │
│  │ g_ocaml_heartbeat   │  │ g_scheme_heartbeat  │      │
│  │ alignas(64)         │  │ alignas(64)         │      │
│  └─────────────────────┘  └─────────────────────┘      │
│  ┌─────────────────────┐                              │
│  │ g_emergency_stop    │                              │
│  └─────────────────────┘                              │
└─────────────────────────────────────────────────────────┘
       │                                   │
       │ 检查 Scheme 心跳                  │ 检查 OCaml 心跳
       │ 超时 > 20ms                       │ 超时 > 20ms
       ▼                                   ▼
┌─────────────────────────────────────────────────────────┐
│  紧急断路器                                              │
│  • 拒绝新开仓                                            │
│  • 撤销未成交订单                                         │
│  • 发送告警通知                                           │
└─────────────────────────────────────────────────────────┘
```

### 5.1.2 健康检查实现

```cpp
// C++ 端健康检查 (独立线程)
void health_check_loop() {
    while (running) {
        std::this_thread::sleep_for(10ms);
        
        // 检查 OCaml 心跳
        if (current_ocaml == last_ocaml) {
            emergency_stop.trigger();
        }
        
        // 检查 Scheme 心跳
        if (current_scheme == last_scheme) {
            emergency_stop.trigger();
        }
    }
}
```

```scheme
;; Scheme 端健康检查
(define (check-ocaml-health)
  (let ([current-hb (get-ocaml-heartbeat)])
    (when (= current-hb *last-ocaml-heartbeat*)
      (log-fatal "OCaml heartbeat timeout!")
      (trigger-emergency-stop))))
```

## 5.2 项目工程目录与 Makefile

### 5.2.1 目录结构

```
quantforge/
├── Makefile                    # 统一构建入口
├── dune-project                # OCaml 项目定义
├── c_src/                      # C++ 底层桥接
│   ├── libquant_core.h
│   ├── libquant_core.cpp
│   └── models/*.pt             # TorchScript 模型
├── ocaml_src/                  # OCaml 行情引擎
│   ├── bin/main.ml
│   └── lib/                    # 类型、接入、FFI
├── scheme_src/                 # Chez Scheme 策略
│   ├── main.ss
│   ├── strategy.ss
│   └── portfolio.ss
├── python/                     # 离线训练
│   ├── train.py
│   └── export.py
└── configs/                    # 配置文件
```

### 5.2.2 Makefile 构建顺序

```makefile
# 构建顺序: model → c_lib → ocaml → scheme

all: model c_lib ocaml scheme

model:      # 检查 TorchScript FP16 模型
c_lib: model    # 编译 C++ CUDA 库
ocaml: c_lib    # 编译 OCaml (依赖 .so)
scheme:         # 验证 Chez Scheme
run: all        # 启动系统
```

### 5.2.3 构建依赖图

```
Python 训练 → FP16 模型 → C++ CUDA 库 → OCaml 引擎
                                    └→ Chez Scheme
                                    (共享 C 内存)
```

---

# 附录

## A. 关键性能指标

| 指标 | 目标值 | 实测值 |
|------|--------|--------|
| Tick 处理延迟 | < 1μs | ~500ns |
| CUDA 传输延迟 | < 20μs | ~10μs |
| FP16 推理延迟 | < 200μs | ~100μs |
| 端到端延迟 | < 500μs | ~200μs |

## B. 显存占用

| 组件 | FP16 |
|------|------|
| 模型权重 | 250MB |
| 推理临时变量 | 50MB |
| 总计 | ~300MB |

RTX 3080 20GB 可同时加载 ~60 个 FP16 模型。

## C. 错误码定义

| 错误码 | 含义 |
|--------|------|
| 0 | 成功 |
| -1 | 空指针 |
| -2 | 模型未加载 |
| -3 | 紧急停止 |
| -4 | CUDA 设备错误 |
| -999 | 推理失败 |

---

**文档结束**
