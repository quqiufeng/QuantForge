#include "libquant_core.h"
#include <torch/script.h>
#include <torch/torch.h>
#include <atomic>
#include <cstring>
#include <iostream>

/* 缓存行对齐的全局特征缓冲区 */
alignas(64) static float g_feature_ring_buffer[FEATURE_BUFFER_SIZE];

/* 原子序列计数器 - 单写单读无锁同步 */
static std::atomic<uint64_t> g_sequence_counter{0};

/* 模型包装结构 */
struct ModelWrapper {
    torch::jit::script::Module module;
    bool loaded;
};

extern "C" {

/* 加载 TorchScript 模型 */
void* load_torch_model(const char* path) {
    if (!path) {
        std::cerr << "[ERROR] Model path is null" << std::endl;
        return nullptr;
    }
    
    try {
        auto* wrapper = new ModelWrapper();
        
        /* 加载模型 */
        wrapper->module = torch::jit::load(path);
        
        /* 设置为推理模式 */
        wrapper->module.eval();
        
        /* 禁用 LibTorch 多线程，避免 GC 抖动 */
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        
        wrapper->loaded = true;
        
        std::cout << "[INFO] Model loaded successfully: " << path << std::endl;
        return static_cast<void*>(wrapper);
        
    } catch (const c10::Error& e) {
        std::cerr << "[ERROR] LibTorch error loading model: " << e.what() << std::endl;
        return nullptr;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception loading model: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "[ERROR] Unknown exception loading model" << std::endl;
        return nullptr;
    }
}

/* 执行推理 - 零拷贝 Tensor 包装 */
float run_inference(void* model_ptr) {
    if (!model_ptr) {
        return -999.0f;
    }
    
    auto* wrapper = static_cast<ModelWrapper*>(model_ptr);
    if (!wrapper->loaded) {
        return -999.0f;
    }
    
    try {
        /* 使用 NoGradGuard 禁用梯度计算 */
        torch::NoGradGuard no_grad;
        
        /* 使用 torch::from_blob 进行零拷贝包装 */
        /* 直接引用全局 C 缓冲区，不复制数据 */
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .requires_grad(false);
        
        torch::Tensor input_tensor = torch::from_blob(
            g_feature_ring_buffer,
            {1, FEATURE_BUFFER_SIZE},  /* batch_size=1 */
            options
        );
        
        /* 执行前向推理 */
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        at::Tensor output = wrapper->module.forward(inputs).toTensor();
        
        /* 提取标量结果 */
        float result = output.item<float>();
        
        return result;
        
    } catch (const c10::Error& e) {
        std::cerr << "[ERROR] LibTorch inference error: " << e.what() << std::endl;
        return -999.0f;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Inference exception: " << e.what() << std::endl;
        return -999.0f;
    } catch (...) {
        /* 绝对不允许异常飞出 C 边界 */
        std::cerr << "[ERROR] Unknown inference exception" << std::endl;
        return -999.0f;
    }
}

/* 释放模型资源 */
void free_torch_model(void* model_ptr) {
    if (model_ptr) {
        auto* wrapper = static_cast<ModelWrapper*>(model_ptr);
        delete wrapper;
        std::cout << "[INFO] Model freed" << std::endl;
    }
}

/* 获取全局特征缓冲区指针 */
float* get_feature_buffer(void) {
    return g_feature_ring_buffer;
}

/* 获取原子序列计数器指针 - 使用静态变量返回地址 */
uint64_t* get_sequence_counter(void) {
    static uint64_t cached_value = 0;
    cached_value = g_sequence_counter.load(std::memory_order_acquire);
    return &cached_value;
}

/* 写入特征到缓冲区并更新序列号 */
void write_features(const float* data, size_t count) {
    if (!data || count == 0 || count > FEATURE_BUFFER_SIZE) {
        return;
    }
    
    /* 写入数据到缓冲区 */
    std::memcpy(g_feature_ring_buffer, data, count * sizeof(float));
    
    /* 内存屏障确保数据写入完成 */
    std::atomic_thread_fence(std::memory_order_release);
    
    /* 递增序列号 */
    g_sequence_counter.fetch_add(1, std::memory_order_release);
}

/* 读取当前序列号 */
uint64_t read_sequence(void) {
    return g_sequence_counter.load(std::memory_order_acquire);
}

} /* extern "C" */
