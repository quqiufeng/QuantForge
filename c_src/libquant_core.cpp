/*
 * ============================================================
 * QuantForge - RTX 3080 20G CUDA 加速实现
 * ============================================================
 * 
 * 技术栈：
 * 1. Pinned Memory (cudaHostAlloc) - 激活 DMA 高速搬运
 * 2. CUDA Streams (non_blocking=true) - 异步非阻塞传输
 * 3. FP16 Half Precision - Tensor Core 加速推理
 * 
 * 数据流：
 * OCaml Lwt → g_feature_ring_buffer (pinned) → 
 * → CUDA Stream 异步 → GPU VRAM → FP16 推理 → 结果
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

/* ============================================================
 * 全局变量定义
 * ============================================================ */

/* 特征缓冲区 - CUDA 固定内存指针 */
float* g_feature_ring_buffer = nullptr;

/* 原子序列计数器 */
alignas(64) static std::atomic<uint64_t> s_sequence_counter{0};
alignas(64) uint64_t g_sequence_counter = 0;

/* 心跳计数器 */
alignas(64) uint64_t g_ocaml_heartbeat = 0;
alignas(64) uint64_t g_scheme_heartbeat = 0;

/* 紧急停止标志 */
alignas(64) uint32_t g_emergency_stop = 0;

/* GPU 状态 */
int g_cuda_available = 0;
int g_cuda_device_id = 0;

/* CUDA Stream 用于异步传输 */
static cudaStream_t s_cuda_stream = nullptr;

/* ============================================================
 * 模型包装结构
 * ============================================================ */
struct ModelWrapper {
    torch::jit::script::Module module;
    bool loaded;
    bool use_cuda;
    bool use_fp16;
};

extern "C" {

/* ============================================================
 * CUDA 初始化
 * ============================================================
 * 
 * 初始化步骤：
 * 1. 检查 CUDA 可用性
 * 2. 创建 CUDA Stream (用于异步传输)
 * 3. 使用 cudaHostAlloc 分配固定内存 (Page-locked)
 * 
 * 固定内存优势：
 * - 激活 GPU DMA 引擎，CPU→GPU 传输速度提升 2-3x
 * - 配合 non_blocking=true 实现真正的异步搬运
 */
int init_cuda_context(void) {
    /* 检查 CUDA 可用性 */
    if (!torch::cuda::is_available()) {
        std::cerr << "[WARN] CUDA not available, falling back to CPU mode" << std::endl;
        g_cuda_available = 0;
        
        /* CPU 模式使用普通内存分配 */
        g_feature_ring_buffer = new(std::nothrow) float[FEATURE_BUFFER_SIZE];
        if (!g_feature_ring_buffer) {
            std::cerr << "[ERROR] Failed to allocate feature buffer" << std::endl;
            return -1;
        }
        std::memset(g_feature_ring_buffer, 0, FEATURE_BUFFER_SIZE * sizeof(float));
        return 0;
    }
    
    /* CUDA 可用 */
    g_cuda_available = 1;
    g_cuda_device_id = c10::cuda::current_device();
    
    std::cout << "[INFO] CUDA Device: " << get_cuda_device_name() << std::endl;
    std::cout << "[INFO] CUDA Memory: " << get_cuda_free_memory() / (1024*1024) << " MB free" << std::endl;
    
    /* 创建 CUDA Stream 用于异步传输 */
    cudaError_t err = cudaStreamCreateWithFlags(&s_cuda_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to create CUDA stream: " 
                  << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    std::cout << "[INFO] CUDA Stream created (non-blocking mode)" << std::endl;
    
    /* 
     * 使用 cudaHostAlloc 分配固定内存 (Pinned Memory / Page-locked)
     * 
     * 关键点：
     * - cudaHostAllocPortable: 所有 CUDA context 可访问
     * - cudaHostAllocMapped: 可获取 GPU 映射地址 (可选)
     * - 固定内存不会被操作系统换页，DMA 可直接访问
     */
    err = cudaHostAlloc(
        (void**)&g_feature_ring_buffer,
        FEATURE_BUFFER_SIZE * sizeof(float),
        cudaHostAllocPortable | cudaHostAllocWriteCombined
    );
    
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaHostAlloc failed: " 
                  << cudaGetErrorString(err) << std::endl;
        
        /* 回退到普通内存 */
        g_feature_ring_buffer = new(std::nothrow) float[FEATURE_BUFFER_SIZE];
        if (!g_feature_ring_buffer) return -1;
    }
    
    std::memset(g_feature_ring_buffer, 0, FEATURE_BUFFER_SIZE * sizeof(float));
    std::cout << "[INFO] Pinned memory allocated: " 
              << FEATURE_BUFFER_SIZE * sizeof(float) << " bytes" << std::endl;
    
    return 0;
}

/* 清理 CUDA 资源 */
void cleanup_cuda_context(void) {
    if (s_cuda_stream) {
        cudaStreamDestroy(s_cuda_stream);
        s_cuda_stream = nullptr;
    }
    
    if (g_feature_ring_buffer) {
        if (g_cuda_available) {
            cudaFreeHost(g_feature_ring_buffer);
        } else {
            delete[] g_feature_ring_buffer;
        }
        g_feature_ring_buffer = nullptr;
    }
    
    std::cout << "[INFO] CUDA resources cleaned up" << std::endl;
}

/* ============================================================
 * LibTorch 模型加载 - GPU 版
 * ============================================================
 * 
 * 加载流程：
 * 1. torch::jit::load 加载 TorchScript 模型
 * 2. model->eval() 设为推理模式
 * 3. model->to(torch::kCUDA) 移至 GPU
 * 4. at::set_num_threads(1) 禁用 CPU 多线程
 */
void* load_torch_model(const char* path) {
    if (!path) {
        std::cerr << "[ERROR] Model path is null" << std::endl;
        return nullptr;
    }
    
    try {
        auto* wrapper = new ModelWrapper();
        wrapper->use_cuda = g_cuda_available;
        wrapper->use_fp16 = g_cuda_available;  /* GPU 模式默认使用 FP16 */
        
        /* 加载模型 */
        wrapper->module = torch::jit::load(path);
        
        /* 设为推理模式 */
        wrapper->module.eval();
        
        /* 禁用 CPU 多线程 */
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        
        /* 移至 GPU */
        if (wrapper->use_cuda) {
            /* 
             * 使用 CUDA Guard 确保在正确的 GPU 设备上
             * RTX 3080 20G 通常为 device 0
             */
            c10::cuda::CUDAGuard guard(g_cuda_device_id);
            wrapper->module.to(torch::kCUDA);
            
            /* 
             * 可选：转换为 FP16 半精度
             * Tensor Core 在 FP16 下算力翻倍
             * 
             * 注意：某些模型可能不支持直接 half()，需要根据模型调整
             */
            try {
                wrapper->module.to(torch::kHalf);
                std::cout << "[INFO] Model converted to FP16 (Tensor Core enabled)" << std::endl;
            } catch (...) {
                std::cout << "[INFO] Model kept in FP32 (FP16 conversion failed)" << std::endl;
                wrapper->use_fp16 = false;
            }
        }
        
        wrapper->loaded = true;
        
        std::cout << "[INFO] Model loaded: " << path << std::endl;
        std::cout << "[INFO] Device: " << (wrapper->use_cuda ? "CUDA" : "CPU") << std::endl;
        std::cout << "[INFO] Precision: " << (wrapper->use_fp16 ? "FP16" : "FP32") << std::endl;
        
        return static_cast<void*>(wrapper);
        
    } catch (const c10::Error& e) {
        std::cerr << "[ERROR] LibTorch: " << e.what() << std::endl;
        return nullptr;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "[ERROR] Unknown exception in model load" << std::endl;
        return nullptr;
    }
}

/* ============================================================
 * 推理函数 - CUDA 异步流 + FP16 版
 * ============================================================
 * 
 * RTX 3080 20G 优化流程：
 * 
 * 1. 从 Pinned Memory 创建 Tensor (零拷贝)
 *    - g_feature_ring_buffer 是 pinned memory
 *    - from_blob 直接包装，不复制数据
 * 
 * 2. 异步推送到 GPU 显存
 *    - .to(kCUDA, non_blocking=true)
 *    - 使用 CUDA Stream 异步传输
 *    - CPU 立即返回，不等待 GPU
 * 
 * 3. FP16 转换 (如果模型支持)
 *    - .to(kHalf) 利用 Tensor Core
 *    - RTX 3080 FP16 算力: 29.8 TFLOPS
 * 
 * 4. 前向推理
 *    - model->forward(input_tensor)
 *    - 结果自动同步回 CPU
 * 
 * 异常处理：
 * - CUDA OOM: 返回 -999.0f
 * - 模型错误: 返回 -999.0f
 * - 所有异常在 C++ 层消化，不飞出 C 边界
 */
float run_inference(void* model_ptr) {
    if (!model_ptr) {
        return -999.0f;
    }
    
    auto* wrapper = static_cast<ModelWrapper*>(model_ptr);
    if (!wrapper->loaded) {
        return -999.0f;
    }
    
    /* 检查紧急停止 */
    if (g_emergency_stop) {
        return -999.0f;
    }
    
    try {
        /* 禁用梯度 */
        torch::NoGradGuard no_grad;
        
        /* 
         * 步骤 1: 从 Pinned Memory 创建零拷贝 Tensor
         * 
         * 注意：使用 pinned memory + CUDA stream 时，
         * from_blob 创建的 Tensor 可以直接用于异步传输
         */
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .requires_grad(false);
        
        torch::Tensor cpu_tensor = torch::from_blob(
            g_feature_ring_buffer,
            {1, FEATURE_BUFFER_SIZE},
            options
        );
        
        /* 
         * 步骤 2: 异步推送到 GPU
         * 
         * non_blocking=true 的关键作用：
         * - CPU 不等待传输完成就立即返回
         * - 传输在 CUDA Stream 后台进行
         * - OCaml 可以立即写入下一个 Tick
         * 
         * 如果模型需要 FP16，同时进行类型转换
         */
        torch::Tensor gpu_tensor;
        
        if (wrapper->use_cuda) {
            c10::cuda::CUDAGuard guard(g_cuda_device_id);
            
            if (wrapper->use_fp16) {
                /* 一步完成: CPU → GPU + FP32 → FP16 */
                gpu_tensor = cpu_tensor.to(
                    torch::Device(torch::kCUDA, g_cuda_device_id),
                    torch::kHalf,
                    /*non_blocking=*/true
                );
            } else {
                /* 仅传输: CPU → GPU */
                gpu_tensor = cpu_tensor.to(
                    torch::Device(torch::kCUDA, g_cuda_device_id),
                    /*non_blocking=*/true
                );
            }
        } else {
            /* CPU 模式：直接使用 */
            gpu_tensor = cpu_tensor;
        }
        
        /* 
         * 步骤 3: 执行前向推理
         * 
         * 注意：forward() 内部会自动同步 CUDA stream
         * 确保异步传输完成后再开始计算
         */
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(gpu_tensor);
        
        at::Tensor output = wrapper->module.forward(inputs).toTensor();
        
        /* 
         * 步骤 4: 获取结果
         * 
         * .item<float>() 会自动从 GPU 同步回 CPU
         */
        float result = output.item<float>();
        
        return result;
        
    } catch (const c10::Error& e) {
        /* LibTorch 错误 (包括 CUDA 错误) */
        std::cerr << "[ERROR] Inference: " << e.what() << std::endl;
        return -999.0f;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Inference: " << e.what() << std::endl;
        return -999.0f;
    } catch (...) {
        /* 绝对不允许异常飞出 C 边界 */
        std::cerr << "[ERROR] Unknown inference exception" << std::endl;
        return -999.0f;
    }
}

void free_torch_model(void* model_ptr) {
    if (model_ptr) {
        auto* wrapper = static_cast<ModelWrapper*>(model_ptr);
        delete wrapper;
        std::cout << "[INFO] Model freed" << std::endl;
    }
}

/* ============================================================
 * OCaml Bigarray 裸指针映射接口
 * ============================================================ */
float* get_feature_buffer_ptr(void) {
    return g_feature_ring_buffer;
}

uint64_t* get_sequence_counter_ptr(void) {
    return &g_sequence_counter;
}

/* ============================================================
 * 无锁写入接口
 * ============================================================ */
void write_features(const float* data, size_t count) {
    if (!data || count == 0 || count > FEATURE_BUFFER_SIZE || !g_feature_ring_buffer) {
        return;
    }
    
    /* 写入 Pinned Memory */
    std::memcpy(g_feature_ring_buffer, data, count * sizeof(float));
    
    /* Release Barrier */
    std::atomic_thread_fence(std::memory_order_release);
    
    /* 递增序列号 */
    uint64_t new_seq = s_sequence_counter.fetch_add(1, std::memory_order_release) + 1;
    g_sequence_counter = new_seq;
}

uint64_t read_sequence(void) {
    return s_sequence_counter.load(std::memory_order_acquire);
}

/* ============================================================
 * 心跳与断路器
 * ============================================================ */
void update_ocaml_heartbeat(void) {
    g_ocaml_heartbeat++;
}

void update_scheme_heartbeat(void) {
    g_scheme_heartbeat++;
}

int check_system_health(void) {
    static uint64_t last_ocaml_heartbeat = 0;
    static uint64_t last_scheme_heartbeat = 0;
    static auto last_check_time = std::chrono::steady_clock::now();
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_check_time
    ).count();
    
    if (elapsed < 10) return 0;
    
    last_check_time = now;
    
    uint64_t current_ocaml = g_ocaml_heartbeat;
    if (current_ocaml == last_ocaml_heartbeat && elapsed > HEARTBEAT_TIMEOUT_MS) {
        std::cerr << "[FATAL] OCaml heartbeat timeout!" << std::endl;
        trigger_emergency_stop();
        return 1;
    }
    last_ocaml_heartbeat = current_ocaml;
    
    uint64_t current_scheme = g_scheme_heartbeat;
    if (current_scheme == last_scheme_heartbeat && elapsed > HEARTBEAT_TIMEOUT_MS) {
        std::cerr << "[FATAL] Scheme heartbeat timeout!" << std::endl;
        trigger_emergency_stop();
        return 2;
    }
    last_scheme_heartbeat = current_scheme;
    
    return 0;
}

void trigger_emergency_stop(void) {
    g_emergency_stop = 1;
    std::cerr << "[ALERT] EMERGENCY STOP TRIGGERED!" << std::endl;
}

int is_emergency_stopped(void) {
    return g_emergency_stop ? 1 : 0;
}

void clear_emergency_stop(void) {
    g_emergency_stop = 0;
    std::cout << "[INFO] Emergency stop cleared" << std::endl;
}

/* ============================================================
 * GPU 信息接口
 * ============================================================ */
int get_cuda_device_count(void) {
    try {
        return torch::cuda::device_count();
    } catch (...) {
        return 0;
    }
}

const char* get_cuda_device_name(void) {
    static char name[256] = "Unknown";
    try {
        if (torch::cuda::is_available()) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, g_cuda_device_id);
            std::strncpy(name, prop.name, sizeof(name) - 1);
        }
    } catch (...) {}
    return name;
}

size_t get_cuda_free_memory(void) {
    try {
        if (torch::cuda::is_available()) {
            size_t free, total;
            cudaMemGetInfo(&free, &total);
            return free;
        }
    } catch (...) {}
    return 0;
}

} /* extern "C" */
