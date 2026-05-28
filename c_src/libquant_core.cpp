/*
 * ============================================================
 * QuantForge - 超低延迟量化交易系统
 * C++ 底层实现 (libquant_core.cpp)
 * ============================================================
 * 
 * 技术栈:
 * - RTX 3080 20G CUDA 加速
 * - LibTorch FP16 半精度推理
 * - Pinned Memory 零拷贝
 * - CUDA Streams 异步传输
 * - 缓存行对齐防止 False Sharing
 * - 内存屏障保证数据一致性
 */

#include "libquant_core.h"

#include <torch/script.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstring>
#include <chrono>
#include <iostream>
#include <memory>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

/* ============================================================
 * 全局状态变量定义
 * ============================================================ */

/* 特征缓冲区 - 使用 CUDA Pinned Memory */
float* g_qf_feature_buffer = nullptr;

/* 缓存行对齐的原子计数器 (防止 False Sharing) */
alignas(64) static std::atomic<uint64_t> s_sequence_atomic{0};
alignas(64) uint64_t g_qf_sequence_counter = 0;

alignas(64) uint64_t g_qf_ocaml_heartbeat = 0;
alignas(64) uint64_t g_qf_scheme_heartbeat = 0;

alignas(64) uint32_t g_qf_emergency_stop = 0;

/* GPU 状态 */
int32_t g_qf_cuda_available = 0;
int32_t g_qf_cuda_device_id = 0;

/* 模型包装结构 */
struct ModelWrapper {
    torch::jit::script::Module module;
    bool loaded;
    bool use_cuda;
    bool use_fp16;
    int device_id;
};

/* 健康检查状态 */
static std::chrono::steady_clock::time_point s_last_health_check;
static uint64_t s_last_ocaml_hb = 0;
static uint64_t s_last_scheme_hb = 0;

/* 设备名称缓冲区 */
static char s_device_name[256] = "Unknown";

/* ============================================================
 * 内部辅助函数
 * ============================================================ */

/* 检查紧急停止 */
static inline bool check_emergency() {
    return g_qf_emergency_stop != 0;
}

/* x86 写屏障 */
static inline void write_barrier() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_sfence();
#elif defined(__aarch64__)
    __dmb(ish);
#endif
}

/* 缓存行预取 */
static inline void prefetch_line(const void* addr) {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
#elif defined(__aarch64__)
    __asm__ volatile("prfm pldl1keep, [%0]" : : "r"(addr));
#endif
}

/* ============================================================
 * 初始化与清理
 * ============================================================ */

extern "C" qf_error_t qf_init(void) {
    /* 检查 CUDA 可用性 */
    if (torch::cuda::is_available()) {
        g_qf_cuda_available = 1;
        g_qf_cuda_device_id = c10::cuda::current_device();
        
        /* 获取设备名称 */
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, g_qf_cuda_device_id);
        std::strncpy(s_device_name, prop.name, sizeof(s_device_name) - 1);
        
        std::cout << "[INFO] CUDA Device: " << s_device_name << std::endl;
    } else {
        g_qf_cuda_available = 0;
        std::cout << "[WARN] CUDA not available, CPU mode" << std::endl;
    }
    
    /* 分配特征缓冲区 */
    if (g_qf_cuda_available) {
        /* 使用 cudaHostAlloc 分配 Pinned Memory */
        cudaError_t err = cudaHostAlloc(
            reinterpret_cast<void**>(&g_qf_feature_buffer),
            QF_FEATURE_BUFFER_SIZE * sizeof(float),
            cudaHostAllocPortable | cudaHostAllocWriteCombined
        );
        
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] cudaHostAlloc failed: " 
                      << cudaGetErrorString(err) << std::endl;
            /* 回退到普通内存 */
            g_qf_feature_buffer = new(std::nothrow) float[QF_FEATURE_BUFFER_SIZE];
            if (!g_qf_feature_buffer) return QF_ERROR_CUDA_DEVICE;
        } else {
            std::cout << "[INFO] Pinned memory allocated: " 
                      << QF_FEATURE_BUFFER_SIZE * sizeof(float) << " bytes" << std::endl;
        }
    } else {
        /* CPU 模式使用普通内存 */
        g_qf_feature_buffer = new(std::nothrow) float[QF_FEATURE_BUFFER_SIZE];
        if (!g_qf_feature_buffer) return QF_ERROR_NULL_PTR;
    }
    
    /* 初始化缓冲区 */
    std::memset(g_qf_feature_buffer, 0, QF_FEATURE_BUFFER_SIZE * sizeof(float));
    
    /* 初始化时间戳 */
    s_last_health_check = std::chrono::steady_clock::now();
    
    std::cout << "[INFO] QuantForge initialized" << std::endl;
    return QF_OK;
}

extern "C" void qf_cleanup(void) {
    /* 释放特征缓冲区 */
    if (g_qf_feature_buffer) {
        if (g_qf_cuda_available) {
            cudaFreeHost(g_qf_feature_buffer);
        } else {
            delete[] g_qf_feature_buffer;
        }
        g_qf_feature_buffer = nullptr;
    }
    
    std::cout << "[INFO] QuantForge cleaned up" << std::endl;
}

/* ============================================================
 * 缓冲区操作
 * ============================================================ */

extern "C" qf_buffer_t qf_get_buffer_ptr(void) {
    return reinterpret_cast<qf_buffer_t>(g_qf_feature_buffer);
}

extern "C" qf_buffer_t qf_get_sequence_ptr(void) {
    return reinterpret_cast<qf_buffer_t>(&g_qf_sequence_counter);
}

extern "C" qf_sequence_t qf_read_sequence(void) {
    /* Acquire 语义读取 */
    return s_sequence_atomic.load(std::memory_order_acquire);
}

extern "C" void qf_write_features(const float* data, size_t count) {
    if (!data || count == 0 || count > QF_FEATURE_BUFFER_SIZE || !g_qf_feature_buffer) {
        return;
    }
    
    /* 1. 写入特征数据到 Pinned Memory */
    std::memcpy(g_qf_feature_buffer, data, count * sizeof(float));
    
    /* 2. 写内存屏障 - 确保 memcpy 完成 */
    std::atomic_thread_fence(std::memory_order_release);
    
    /* 3. 更新序列号 (Release 语义) */
    uint64_t new_seq = s_sequence_atomic.fetch_add(1, std::memory_order_release) + 1;
    g_qf_sequence_counter = new_seq;
    
    /* 4. 硬件写屏障 */
    write_barrier();
}

extern "C" qf_error_t qf_read_features_safe(float* output, size_t count, qf_sequence_t* out_seq) {
    if (!output || !out_seq || count == 0 || count > QF_FEATURE_BUFFER_SIZE) {
        return QF_ERROR_NULL_PTR;
    }
    
    /* 1. 读取序列号 (Acquire) */
    qf_sequence_t seq = s_sequence_atomic.load(std::memory_order_acquire);
    
    /* 2. 复制数据 */
    std::memcpy(output, g_qf_feature_buffer, count * sizeof(float));
    
    /* 3. 再次读取序列号验证 */
    qf_sequence_t seq_after = s_sequence_atomic.load(std::memory_order_relaxed);
    
    /* 4. 检查一致性 */
    if (seq != seq_after) {
        return QF_ERROR_INFERENCE;
    }
    
    *out_seq = seq;
    return QF_OK;
}

/* ============================================================
 * 模型管理
 * ============================================================ */

extern "C" qf_model_t qf_load_model(const char* path) {
    if (!path) {
        std::cerr << "[ERROR] Model path is null" << std::endl;
        return 0;
    }
    
    try {
        auto* wrapper = new ModelWrapper();
        wrapper->use_cuda = g_qf_cuda_available;
        wrapper->use_fp16 = g_qf_cuda_available;
        wrapper->device_id = g_qf_cuda_device_id;
        
        /* 加载 TorchScript 模型 */
        wrapper->module = torch::jit::load(path);
        
        /* 设置为推理模式 */
        wrapper->module.eval();
        
        /* 禁用多线程 */
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        
        /* 移至 GPU */
        if (wrapper->use_cuda) {
            c10::cuda::CUDAGuard guard(wrapper->device_id);
            wrapper->module.to(torch::kCUDA);
            
            /* 转换为 FP16 */
            try {
                wrapper->module.to(torch::kHalf);
                std::cout << "[INFO] Model converted to FP16" << std::endl;
            } catch (...) {
                std::cout << "[WARN] FP16 conversion failed, using FP32" << std::endl;
                wrapper->use_fp16 = false;
            }
        }
        
        wrapper->loaded = true;
        
        std::cout << "[INFO] Model loaded: " << path << std::endl;
        std::cout << "[INFO] Device: " << (wrapper->use_cuda ? "CUDA" : "CPU") << std::endl;
        
        return reinterpret_cast<qf_model_t>(wrapper);
        
    } catch (const c10::Error& e) {
        std::cerr << "[ERROR] LibTorch: " << e.what() << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 0;
    } catch (...) {
        std::cerr << "[ERROR] Unknown exception loading model" << std::endl;
        return 0;
    }
}

extern "C" void qf_free_model(qf_model_t model) {
    if (model != 0) {
        auto* wrapper = reinterpret_cast<ModelWrapper*>(model);
        delete wrapper;
        std::cout << "[INFO] Model freed" << std::endl;
    }
}

extern "C" qf_error_t qf_is_model_valid(qf_model_t model) {
    if (model == 0) return QF_ERROR_NULL_PTR;
    auto* wrapper = reinterpret_cast<ModelWrapper*>(model);
    return wrapper->loaded ? QF_OK : QF_ERROR_MODEL_INVALID;
}

/* ============================================================
 * 推理接口
 * ============================================================ */

extern "C" float qf_run_inference(qf_model_t model) {
    float trend, risk, vol;
    qf_error_t err = qf_run_inference_multi(model, &trend, &risk, &vol);
    if (err != QF_OK) return static_cast<float>(err);
    return risk;
}

extern "C" qf_error_t qf_run_inference_multi(
    qf_model_t model,
    float* out_trend,
    float* out_risk,
    float* out_vol
) {
    /* 参数检查 */
    if (model == 0) return QF_ERROR_NULL_PTR;
    if (!out_trend || !out_risk || !out_vol) return QF_ERROR_NULL_PTR;
    
    auto* wrapper = reinterpret_cast<ModelWrapper*>(model);
    if (!wrapper->loaded) return QF_ERROR_MODEL_LOAD;
    
    /* 检查紧急停止 */
    if (check_emergency()) return QF_ERROR_EMERGENCY;
    
    try {
        /* 禁用梯度 */
        torch::NoGradGuard no_grad;
        
        /* GPU Guard */
        if (wrapper->use_cuda) {
            c10::cuda::CUDAGuard guard(wrapper->device_id);
        }
        
        /* ===== 步骤 1: 从 Pinned Memory 创建零拷贝 Tensor ===== */
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU)
            .requires_grad(false);
        
        torch::Tensor cpu_view = torch::from_blob(
            g_qf_feature_buffer,
            {1, QF_FEATURE_BUFFER_SIZE},
            options
        );
        
        /* ===== 步骤 2: 克隆快照防止覆写 ===== */
        torch::Tensor cpu_snapshot = cpu_view.clone();
        
        /* ===== 步骤 3: 异步推送到 GPU + FP16 ===== */
        torch::Tensor input_tensor;
        
        if (wrapper->use_cuda) {
            if (wrapper->use_fp16) {
                input_tensor = cpu_snapshot.to(
                    torch::Device(torch::kCUDA, wrapper->device_id),
                    torch::kHalf,
                    /*non_blocking=*/true
                );
            } else {
                input_tensor = cpu_snapshot.to(
                    torch::Device(torch::kCUDA, wrapper->device_id),
                    /*non_blocking=*/true
                );
            }
        } else {
            input_tensor = cpu_snapshot;
        }
        
        /* ===== 步骤 4: 执行推理 ===== */
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        c10::IValue output = wrapper->module.forward(inputs);
        
        /* ===== 步骤 5: 解包多任务输出 ===== */
        if (output.isTuple()) {
            auto elements = output.toTuple()->elements();
            
            if (elements.size() < 3) {
                std::cerr << "[ERROR] Expected 3 outputs, got " 
                          << elements.size() << std::endl;
                return QF_ERROR_INFERENCE;
            }
            
            *out_trend = elements[0].toTensor().item<float>();
            *out_risk  = elements[1].toTensor().item<float>();
            *out_vol   = elements[2].toTensor().item<float>();
        } else if (output.isTensor()) {
            auto tensor = output.toTensor();
            if (tensor.numel() >= 3) {
                auto accessor = tensor.accessor<float, 2>();
                *out_trend = accessor[0][0];
                *out_risk  = accessor[0][1];
                *out_vol   = accessor[0][2];
            }
        } else {
            std::cerr << "[ERROR] Unexpected output type" << std::endl;
            return QF_ERROR_INFERENCE;
        }
        
        return QF_OK;
        
    } catch (const c10::Error& e) {
        std::cerr << "[ERROR] LibTorch inference: " << e.what() << std::endl;
        return QF_ERROR_INFERENCE;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Inference exception: " << e.what() << std::endl;
        return QF_ERROR_INFERENCE;
    } catch (...) {
        std::cerr << "[ERROR] Unknown inference exception" << std::endl;
        return QF_ERROR_INFERENCE;
    }
}

/* ============================================================
 * 心跳与健康检查
 * ============================================================ */

extern "C" void qf_update_ocaml_heartbeat(void) {
    g_qf_ocaml_heartbeat++;
}

extern "C" void qf_update_scheme_heartbeat(void) {
    g_qf_scheme_heartbeat++;
}

extern "C" uint64_t qf_get_ocaml_heartbeat(void) {
    return g_qf_ocaml_heartbeat;
}

extern "C" uint64_t qf_get_scheme_heartbeat(void) {
    return g_qf_scheme_heartbeat;
}

extern "C" int32_t qf_check_health(void) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - s_last_health_check
    ).count();
    
    if (elapsed < 10) return 0;
    
    s_last_health_check = now;
    
    /* 检查 OCaml 心跳 */
    uint64_t current_ocaml = g_qf_ocaml_heartbeat;
    if (current_ocaml == s_last_ocaml_hb && elapsed > QF_HEARTBEAT_TIMEOUT_MS) {
        std::cerr << "[FATAL] OCaml heartbeat timeout!" << std::endl;
        qf_emergency_stop();
        return 1;
    }
    s_last_ocaml_hb = current_ocaml;
    
    /* 检查 Scheme 心跳 */
    uint64_t current_scheme = g_qf_scheme_heartbeat;
    if (current_scheme == s_last_scheme_hb && elapsed > QF_HEARTBEAT_TIMEOUT_MS) {
        std::cerr << "[FATAL] Scheme heartbeat timeout!" << std::endl;
        qf_emergency_stop();
        return 2;
    }
    s_last_scheme_hb = current_scheme;
    
    return 0;
}

/* ============================================================
 * 紧急停止控制
 * ============================================================ */

extern "C" void qf_emergency_stop(void) {
    g_qf_emergency_stop = 1;
    std::cerr << "[ALERT] EMERGENCY STOP TRIGGERED!" << std::endl;
}

extern "C" int32_t qf_is_emergency(void) {
    return g_qf_emergency_stop ? 1 : 0;
}

extern "C" void qf_clear_emergency(void) {
    g_qf_emergency_stop = 0;
    std::cout << "[INFO] Emergency stop cleared" << std::endl;
}

/* ============================================================
 * GPU 信息
 * ============================================================ */

extern "C" int32_t qf_cuda_available(void) {
    return g_qf_cuda_available;
}

extern "C" int32_t qf_cuda_device_count(void) {
    try {
        return static_cast<int32_t>(torch::cuda::device_count());
    } catch (...) {
        return 0;
    }
}

extern "C" const char* qf_cuda_device_name(void) {
    return s_device_name;
}

extern "C" uint64_t qf_cuda_free_memory(void) {
    try {
        if (g_qf_cuda_available) {
            size_t free, total;
            cudaMemGetInfo(&free, &total);
            return static_cast<uint64_t>(free);
        }
    } catch (...) {}
    return 0;
}

extern "C" void qf_prefetch_buffer(void) {
    if (!g_qf_feature_buffer) return;
    
    for (size_t i = 0; i < QF_FEATURE_BUFFER_SIZE * sizeof(float); i += 64) {
        prefetch_line(
            reinterpret_cast<const char*>(g_qf_feature_buffer) + i
        );
    }
}
