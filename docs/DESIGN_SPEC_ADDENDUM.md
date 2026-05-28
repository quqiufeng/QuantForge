# QuantForge 实现规格补充修正案

**版本**: 1.1  
**日期**: 2026-05-28  
**关联文档**: DESIGN_SPEC.md v1.0  

---

## 修正条款 1：C/C++ 导出接口 ABI 稳定性规范

### 1.1 问题分析

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ABI 不稳定性风险                                    │
└─────────────────────────────────────────────────────────────────────────┘

当前实现:
  void* load_torch_model(const char* path)
  float run_inference(void* model_ptr)

风险:
1. void* 在 32/64 位系统长度不同
2. C++ 名称修饰 (Name Mangling) 导致符号不可预测
3. Chez Scheme foreign-procedure 需要精确的符号名
4. OCaml Ctypes 需要明确的类型声明
```

### 1.2 修正规范

**规则 1.2.1**: 所有指针类型显式定义为 `intptr_t`/`uintptr_t`

```cpp
// c_src/libquant_core.h

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// 类型别名 - 确保跨平台一致性
typedef intptr_t  qf_model_t;    // 模型句柄
typedef intptr_t  qf_buffer_t;   // 缓冲区句柄
typedef uint64_t  qf_sequence_t; // 序列号

// 错误码
typedef int32_t   qf_error_t;
#define QF_OK                   0
#define QF_ERROR_NULL_PTR      -1
#define QF_ERROR_MODEL_LOAD    -2
#define QF_ERROR_INFERENCE     -999

#ifdef __cplusplus
}
#endif
```

**规则 1.2.2**: 统一函数命名前缀 `qf_`

```cpp
// c_src/libquant_core.h - 完整接口定义

#ifdef __cplusplus
extern "C" {
#endif

// ===== 模型管理 =====
qf_model_t qf_load_model(const char* path);
void qf_free_model(qf_model_t model);
qf_error_t qf_is_model_valid(qf_model_t model);

// ===== 推理 =====
float qf_run_inference(qf_model_t model);
qf_error_t qf_run_inference_multi(
    qf_model_t model,
    float* out_trend,    // [输出] 趋势预测
    float* out_risk,     // [输出] 风险概率
    float* out_vol       // [输出] 波动率
);

// ===== 缓冲区 =====
qf_buffer_t qf_get_buffer_ptr(void);
qf_sequence_t qf_read_sequence(void);
void qf_write_features(const float* data, uintptr_t count);

// ===== 心跳 =====
void qf_update_ocaml_heartbeat(void);
void qf_update_scheme_heartbeat(void);
int32_t qf_check_health(void);
void qf_emergency_stop(void);
int32_t qf_is_emergency(void);

#ifdef __cplusplus
}
#endif
```

### 1.3 Chez Scheme FFI 映射

```scheme
;; scheme_src/ffi.ss

(load-shared-object "./libquant_core.so")

;; ===== 类型映射 =====
;; C 类型          Scheme 类型
;; intptr_t    →   iptr
;; uintptr_t   →   uptr
;; int32_t     →   int
;; uint64_t    →   unsigned-64
;; float       →   float
;; const char* →   string

;; ===== 模型管理 =====
(define qf-load-model
  (foreign-procedure "qf_load_model" (string) iptr))

(define qf-free-model
  (foreign-procedure "qf_free_model" (iptr) void))

(define qf-is-model-valid
  (foreign-procedure "qf_is_model_valid" (iptr) int))

;; ===== 推理 =====
(define qf-run-inference
  (foreign-procedure "qf_run_inference" (iptr) float))

;; 多输出推理 - 使用 Scheme 向量接收
(define qf-run-inference-multi
  (foreign-procedure "qf_run_inference_multi" 
                     (iptr (* float) (* float) (* float))
                     int))

;; ===== 缓冲区 =====
(define qf-get-buffer-ptr
  (foreign-procedure "qf_get_buffer_ptr" () uptr))

(define qf-read-sequence
  (foreign-procedure "qf_read_sequence" () unsigned-64))

(define qf-write-features
  (foreign-procedure "qf_write_features" ((* float) uptr) void))

;; ===== 心跳 =====
(define qf-update-ocaml-heartbeat
  (foreign-procedure "qf_update_ocaml_heartbeat" () void))

(define qf-update-scheme-heartbeat
  (foreign-procedure "qf_update_scheme_heartbeat" () void))

(define qf-check-health
  (foreign-procedure "qf_check_health" () int))

(define qf-emergency-stop
  (foreign-procedure "qf_emergency_stop" () void))

(define qf-is-emergency
  (foreign-procedure "qf_is_emergency" () int))
```

### 1.4 OCaml 4 Ctypes 映射

```ocaml
(* ocaml_src/lib/c_ffi.ml *)

open Ctypes
open Foreign

(* 类型定义 *)
type qf_model = nativeint   (* intptr_t *)
type qf_buffer = nativeint  (* intptr_t *)
type qf_sequence = int64    (* uint64_t *)

(* 模型管理 *)
let qf_load_model =
  foreign "qf_load_model" 
    (string @-> returning nativeint)

let qf_free_model =
  foreign "qf_free_model"
    (nativeint @-> returning void)

(* 推理 *)
let qf_run_inference =
  foreign "qf_run_inference"
    (nativeint @-> returning float)

(* 缓冲区 *)
let qf_get_buffer_ptr =
  foreign "qf_get_buffer_ptr"
    (void @-> returning nativeint)

let qf_read_sequence =
  foreign "qf_read_sequence"
    (void @-> returning uint64_t)
```

---

## 修正条款 2：LibTorch 多任务输出张量解包规范

### 2.1 问题分析

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Multi-Task Head 输出结构                              │
└─────────────────────────────────────────────────────────────────────────┘

model.forward() 返回类型: c10::IValue

IValue 内部结构 (PyTorch trace 后):
┌─────────────────────────────────────────┐
│  IValue (tuple)                         │
│  ├── elements()[0] → Tensor (Trend)     │
│  ├── elements()[1] → Tensor (Risk)      │
│  └── elements()[2] → Tensor (Vol)       │
└─────────────────────────────────────────┘

错误代码:
  output.toTensor().item<float>()  // 会抛异常!
  // IValue 不是 Tensor，不能直接 toTensor()
```

### 2.2 修正实现

```cpp
// c_src/libquant_core.cpp

// 单输出推理 (兼容旧接口)
extern "C" float qf_run_inference(qf_model_t model_handle) {
    float trend, risk, vol;
    qf_error_t err = qf_run_inference_multi(model_handle, &trend, &risk, &vol);
    if (err != QF_OK) return static_cast<float>(err);
    return risk;  // 默认返回风险概率
}

// 多输出推理 (推荐接口)
extern "C" qf_error_t qf_run_inference_multi(
    qf_model_t model_handle,
    float* out_trend,
    float* out_risk,
    float* out_vol
) {
    // 参数检查
    if (model_handle == 0) return QF_ERROR_NULL_PTR;
    if (!out_trend || !out_risk || !out_vol) return QF_ERROR_NULL_PTR;
    
    auto* wrapper = reinterpret_cast<ModelWrapper*>(model_handle);
    if (!wrapper->loaded) return QF_ERROR_MODEL_LOAD;
    
    auto& state = GlobalSharedState::instance();
    if (state.emergency_stop.is_triggered()) return QF_ERROR_INFERENCE;
    
    try {
        torch::NoGradGuard no_grad;
        c10::cuda::CUDAGuard device_guard(0);
        
        // 1. 零拷贝 + 快照
        auto cpu_view = torch::from_blob(
            state.feature_buffer.data,
            {1, FEATURE_BUFFER_SIZE}
        );
        auto cpu_snapshot = cpu_view.clone();
        
        // 2. 异步推送到 GPU + FP16
        auto gpu_tensor = cpu_snapshot.to(
            torch::kCUDA, torch::kHalf, true
        );
        
        // 3. 执行推理
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(gpu_tensor);
        
        c10::IValue output = wrapper->module.forward(inputs);
        
        // ===== 关键: 正确解包多任务输出 =====
        
        // 方案 A: 如果模型返回 tuple (trace 默认行为)
        if (output.isTuple()) {
            auto elements = output.toTuple()->elements();
            
            // 验证输出数量
            if (elements.size() < 3) {
                std::cerr << "[ERROR] Expected 3 outputs, got " 
                          << elements.size() << std::endl;
                return QF_ERROR_INFERENCE;
            }
            
            // 提取每个头的输出
            *out_trend = elements[0].toTensor().item<float>();
            *out_risk  = elements[1].toTensor().item<float>();
            *out_vol   = elements[2].toTensor().item<float>();
        }
        // 方案 B: 如果模型返回 dict (需要在 export 时配置)
        else if (output.isGenericDict()) {
            auto dict = output.toGenericDict();
            
            *out_trend = dict.at("trend").toTensor().item<float>();
            *out_risk  = dict.at("risk").toTensor().item<float>();
            *out_vol   = dict.at("volatility").toTensor().item<float>();
        }
        // 方案 C: 单 Tensor 输出 (降级处理)
        else if (output.isTensor()) {
            auto tensor = output.toTensor();
            if (tensor.numel() >= 3) {
                auto accessor = tensor.accessor<float, 2>();
                *out_trend = accessor[0][0];
                *out_risk  = accessor[0][1];
                *out_vol   = accessor[0][2];
            }
        }
        else {
            std::cerr << "[ERROR] Unexpected output type" << std::endl;
            return QF_ERROR_INFERENCE;
        }
        
        return QF_OK;
        
    } catch (const c10::Error& e) {
        std::cerr << "[ERROR] LibTorch: " << e.what() << std::endl;
        return QF_ERROR_INFERENCE;
    } catch (...) {
        std::cerr << "[ERROR] Unknown inference exception" << std::endl;
        return QF_ERROR_INFERENCE;
    }
}
```

### 2.3 Python 端导出规范

```python
# python/export.py

class QuantForgeModel(nn.Module):
    """确保 trace 时返回 tuple"""
    
    def forward(self, x):
        H = self.backbone(x)
        trend = self.head_trend(H)
        risk = self.head_risk(H)
        vol = self.head_vol(H)
        # 显式返回 tuple
        return (trend, risk, vol)

# Trace 时提供示例输出
model = QuantForgeModel()
example_input = torch.randn(1, 128, 64)

# 确保 trace 正确捕获输出结构
with torch.no_grad():
    example_output = model(example_input)
    print(f"Output type: {type(example_output)}")
    print(f"Output[0] shape: {example_output[0].shape}")  # (1, 1)
    print(f"Output[1] shape: {example_output[1].shape}")  # (1, 1)
    print(f"Output[2] shape: {example_output[2].shape}")  # (1, 1)

traced = torch.jit.trace(model, example_input)

# 验证 trace 结果
with torch.no_grad():
    traced_output = traced(example_input)
    assert isinstance(traced_output, tuple)
    assert len(traced_output) == 3
```

---

## 修正条款 3：Chez Scheme 零分配轮询与 GC 协同

### 3.1 问题分析

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Scheme GC 延迟抖动风险                                │
└─────────────────────────────────────────────────────────────────────────┘

传统实现:
(define (polling-loop)
  (let ([data (cons seq (cons price '()))])  ; ← 每次循环分配!
    (process data)
    (sleep 0.001)
    (polling-loop)))

问题:
1. cons/list 每次调用都分配堆内存
2. 高频轮询 (1ms) 导致大量临时对象
3. 触发 Major GC → 延迟毛刺 10-100ms
4. 系统响应时间不可预测
```

### 3.2 零分配轮询规范

**规则 3.2.1**: 禁止在轮询循环中使用堆分配操作符

```scheme
;; ===== 禁止列表 =====
;; cons, list, vector, make-vector, string, make-string
;; append, map, filter, reverse
;; any lambda that captures free variables (closure allocation)

;; ===== 允许列表 =====
;; let, let*, letrec (无捕获时)
;; if, cond, case, and, or
;; +, -, *, /, =, <, >, car, cdr
;; set! (修改已有变量)
;; foreign-procedure 调用
```

**规则 3.2.2**: 使用预分配的固定大小存储

```scheme
;; scheme_src/pool.ss

;; 预分配的网格订单存储区 (固定大小，永不增长)
(define GRID-POOL-SIZE 100)

;; 使用 Scheme vector 预分配 (程序启动时分配一次)
(define grid-orders (make-vector GRID-POOL-SIZE))

;; 初始化填充
(define (init-grid-orders!)
  (do ([i 0 (+ i 1)])
      ((= i GRID-POOL-SIZE))
    (vector-set! grid-orders i 
                 (vector 0.0 0.0 #f 0))))  ; [price, qty, side, status]

;; 当前使用的网格数量
(define grid-count 0)

;; ===== 零分配更新操作 =====
(define (update-grid-order! index price qty side status)
  (let ([order (vector-ref grid-orders index)])
    (vector-set! order 0 price)
    (vector-set! order 1 qty)
    (vector-set! order 2 side)
    (vector-set! order 3 status)))

(define (get-grid-price index)
  (vector-ref (vector-ref grid-orders index) 0))

(define (get-grid-qty index)
  (vector-ref (vector-ref grid-orders index) 1))
```

### 3.3 零分配轮询循环实现

```scheme
;; scheme_src/main.ss

;; 预分配的状态变量 (程序启动时)
(define last-sequence 0)
(define current-trend 0.0)
(define current-risk 0.0)
(define current-vol 0.0)

;; 使用 letrec 优化的轮询循环 (无闭包分配)
(define (polling-loop)
  ;; 使用 let 绑定局部变量 (栈分配，非堆分配)
  (let ([seq (qf-read-sequence)])
    ;; 检查新数据
    (when (> seq last-sequence)
      ;; 更新序列号
      (set! last-sequence seq)
      
      ;; 执行推理 (C 堆分配，不影响 Scheme GC)
      (let ([err (qf-run-inference-multi 
                   model-handle 
                   (foreign-ref 'float trend-ptr 0)
                   (foreign-ref 'float risk-ptr 0)
                   (foreign-ref 'float vol-ptr 0))])
        (when (= err 0)
          ;; 更新状态变量 (无分配)
          (set! current-trend (foreign-ref 'float trend-ptr 0))
          (set! current-risk (foreign-ref 'float risk-ptr 0))
          (set! current-vol (foreign-ref 'float vol-ptr 0))
          
          ;; 处理信号 (零分配版本)
          (process-signal-zero-alloc))))
    
    ;; 更新心跳 (无分配)
    (qf-update-scheme-heartbeat)
    
    ;; 尾递归继续 (无栈溢出)
    (polling-loop)))

;; 零分配信号处理
(define (process-signal-zero-alloc)
  ;; 使用 if/cond 避免分配
  (cond
    [(> current-risk 0.8)
     (qf-emergency-stop)]
    [(> current-trend 0.3)
     ;; 直接修改预分配的网格
     (update-grid-order! 0 (get-current-price) 0.01 'buy 1)]
    [(< current-trend -0.3)
     (update-grid-order! 0 (get-current-price) 0.01 'sell 1)]
    [else
     ;; 无操作
     #f]))
```

### 3.4 GC 压力监控

```scheme
;; 调试用: 监控 GC 统计
(define (report-gc-stats)
  (let ([stats (statistics)])
    (printf "GC count: ~a\n" (statistics-gc-count stats))
    (printf "GC time: ~a ms\n" (statistics-gc-time stats))
    (printf "Alloc rate: ~a bytes/sec\n" 
            (/ (statistics-bytes-allocated stats)
               (/ (statistics-gc-time stats) 1000)))))

;; 在开发阶段定期调用
;; (每 10000 次轮询报告一次)
(define poll-counter 0)
(define (maybe-report-stats)
  (set! poll-counter (+ poll-counter 1))
  (when (= (mod poll-counter 10000) 0)
    (report-gc-stats)))
```

---

## 修正条款 4：OCaml 4 内存屏障与缓存一致性

### 4.1 问题分析

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CPU 指令重排导致的数据不一致                           │
└─────────────────────────────────────────────────────────────────────────┘

OCaml Core (写入端):
  1. buffer[0] = bid     ; 特征数据写入
  2. buffer[1] = ask
  3. ... 
  4. g_sequence++        ; 序列号更新

CPU 可能重排为:
  4. g_sequence++        ; 序列号先更新!
  1. buffer[0] = bid     ; 特征数据后写入
  2. buffer[1] = ask
  3. ...

Scheme Core (读取端):
  if (g_sequence > last_seq):
    read buffer[0]  ; ← 读到旧数据! (Dirty Read)
```

### 4.2 内存屏障规范

**规则 4.2.1**: C++ 端 Release/Acquire 屏障

```cpp
// c_src/libquant_core.cpp

// 写入端 (OCaml 调用)
extern "C" void qf_write_features(const float* data, uintptr_t count) {
    auto& state = GlobalSharedState::instance();
    
    if (!data || count == 0 || count > FEATURE_BUFFER_SIZE) return;
    
    // 1. 写入特征数据
    std::memcpy(state.feature_buffer.data, data, count * sizeof(float));
    
    // 2. 写内存屏障 (Release)
    //    确保 memcpy 的所有写入在序列号更新前对其他核心可见
    std::atomic_thread_fence(std::memory_order_release);
    
    // 3. 更新序列号 (Release 语义)
    state.sequence_counter.value.fetch_add(1, std::memory_order_release);
    
    // 4. 额外的硬件屏障 (x86 _mm_sfence 或 ARM __dmb)
    #if defined(__x86_64__) || defined(_M_X64)
        _mm_sfence();  // Store Fence - 确保所有存储完成
    #elif defined(__aarch64__)
        __dmb(ish);    // Data Memory Barrier - Inner Shareable
    #endif
}

// 读取端 (Scheme 调用)
extern "C" qf_sequence_t qf_read_sequence(void) {
    auto& state = GlobalSharedState::instance();
    
    // Acquire 屏障 (隐含在 load 中)
    return state.sequence_counter.value.load(std::memory_order_acquire);
}

// 安全读取特征 (带屏障)
extern "C" qf_error_t qf_read_features_safe(
    float* output, 
    uintptr_t count,
    qf_sequence_t* out_seq
) {
    auto& state = GlobalSharedState::instance();
    
    // 1. 读取序列号 (Acquire)
    qf_sequence_t seq = state.sequence_counter.value.load(
        std::memory_order_acquire
    );
    
    // 2. 读取数据
    std::memcpy(output, state.feature_buffer.data, count * sizeof(float));
    
    // 3. 再次读取序列号确认 (防止读取过程中被覆写)
    qf_sequence_t seq_after = state.sequence_counter.value.load(
        std::memory_order_relaxed
    );
    
    // 4. 验证数据一致性
    if (seq != seq_after) {
        // 读取过程中数据被覆写，重试或返回错误
        return QF_ERROR_INFERENCE;
    }
    
    *out_seq = seq;
    return QF_OK;
}
```

**规则 4.2.2**: OCaml 端 C FFI 内存屏障封装

```ocaml
(* ocaml_src/lib/c_ffi.ml *)

open Ctypes
open Foreign

(* 写入特征 (带内存屏障) *)
let qf_write_features =
  foreign "qf_write_features"
    (ptr float @-> size_t @-> returning void)

(* 
 * OCaml 端的写入函数
 * 注意: Bigarray.set 本身不保证内存顺序
 * 必须通过 C 函数的内存屏障来保证
 *)
let write_tick_features (buffer : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t) 
                         (tick : tick) =
  (* 1. 直接写入 Bigarray (零拷贝) *)
  Bigarray.Array1.set buffer 0 tick.bid;
  Bigarray.Array1.set buffer 1 tick.ask;
  Bigarray.Array1.set buffer 2 tick.bid_qty;
  Bigarray.Array1.set buffer 3 tick.ask_qty;
  Bigarray.Array1.set buffer 4 tick.last_price;
  Bigarray.Array1.set buffer 5 (Int64.to_float tick.timestamp);
  Bigarray.Array1.set buffer 6 (Int64.to_float tick.sequence);
  
  (* 2. 调用 C 函数更新序列号 (包含内存屏障) *)
  (*    这确保所有 Bigarray.set 操作在序列号更新前完成 *)
  let count = Unsigned.Size_t.of_int 7 in
  let buf_ptr = Ctypes.bigarray_start Ctypes.array1 buffer in
  qf_write_features buf_ptr count
  
  (* 注意: 不能直接在 OCaml 中更新序列号 *)
  (* 必须通过 C 函数的内存屏障来保证顺序 *)
```

### 4.3 缓存行预取优化

```cpp
// c_src/libquant_core.cpp

// 可选: 缓存行预取 (高级优化)
extern "C" void qf_prefetch_buffer(void) {
    auto& state = GlobalSharedState::instance();
    
    // 预取特征缓冲区到 L1/L2 缓存
    #if defined(__x86_64__) || defined(_M_X64)
        // Intel/AMD: _mm_prefetch
        for (size_t i = 0; i < FEATURE_BUFFER_SIZE * sizeof(float); i += 64) {
            _mm_prefetch(
                reinterpret_cast<const char*>(state.feature_buffer.data) + i,
                _MM_HINT_T0  // 预取到 L1
            );
        }
    #elif defined(__aarch64__)
        // ARM: PRFM
        for (size_t i = 0; i < FEATURE_BUFFER_SIZE * sizeof(float); i += 64) {
            __asm__ volatile(
                "prfm pldl1keep, [%0]"
                : : "r" (reinterpret_cast<const char*>(state.feature_buffer.data) + i)
            );
        }
    #endif
}
```

### 4.4 完整的读写同步协议

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    完整的无锁读写同步协议                                 │
└─────────────────────────────────────────────────────────────────────────┘

写入端 (OCaml → C):
┌─────────────────────────────────────────┐
│ 1. Bigarray.Array1.set buffer[i] value  │ ← 普通写入
│ 2. ... (多个写入)                        │
│ 3. qf_write_features(buf, count)        │ ← 进入 C
│    ├── memcpy(...)                       │   复制数据
│    ├── atomic_thread_fence(release)      │   写屏障
│    ├── g_sequence.fetch_add(1, release)  │   序列号+1
│    └── _mm_sfence()                      │   硬件屏障
└─────────────────────────────────────────┘

读取端 (Scheme → C):
┌─────────────────────────────────────────┐
│ 1. seq = qf_read_sequence()             │ ← 读取序列号
│    └── load(acquire)                     │   读屏障
│ 2. if (seq > last_seq):                 │
│    ├── qf_read_features_safe(...)       │ ← 读取数据
│    │   ├── memcpy(...)                   │   复制数据
│    │   ├── seq_after = load(relaxed)     │   再次读序列号
│    │   └── if (seq == seq_after) OK      │   验证一致性
│    └── process(data)                    │
└─────────────────────────────────────────┘

内存屏障保证:
- Release (写): 所有之前的写入对后续 Acquire 可见
- Acquire (读): 能看到所有 Release 之前的写入
```

---

## 总结

| 条款 | 问题 | 解决方案 |
|------|------|----------|
| 1. ABI 稳定性 | void* 长度不一致 | intptr_t + qf_ 前缀 |
| 2. Tensor 解包 | IValue 不是 Tensor | toTuple()->elements() |
| 3. GC 协同 | Scheme 堆分配触发 GC | 零分配轮询循环 |
| 4. 内存屏障 | CPU 指令重排导致脏读 | Release/Acquire + sfence |

---

**修正案结束**
