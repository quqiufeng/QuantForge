#ifndef LIBQUANT_CORE_H
#define LIBQUANT_CORE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 缓存行对齐的全局特征缓冲区 (1024 floats) */
#define FEATURE_BUFFER_SIZE 1024

/* 原子序列计数器类型 */
typedef struct {
    uint64_t sequence;
} atomic_counter_t;

/* 加载 TorchScript 模型，返回模型指针，失败返回 NULL */
void* load_torch_model(const char* path);

/* 执行推理，返回预测值，失败返回 -999.0f */
float run_inference(void* model_ptr);

/* 释放模型资源 */
void free_torch_model(void* model_ptr);

/* 获取全局特征缓冲区指针 */
float* get_feature_buffer(void);

/* 获取原子序列计数器指针 */
uint64_t* get_sequence_counter(void);

/* 写入特征到缓冲区并更新序列号 */
void write_features(const float* data, size_t count);

/* 读取当前序列号 */
uint64_t read_sequence(void);

#ifdef __cplusplus
}
#endif

#endif /* LIBQUANT_CORE_H */
