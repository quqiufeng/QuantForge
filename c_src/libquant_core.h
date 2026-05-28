#ifndef LIBQUANT_CORE_H
#define LIBQUANT_CORE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ============================================================
 * QuantForge - RTX 3080 20G CUDA 加速版
 * ============================================================
 */

#define FEATURE_BUFFER_SIZE 1024
#define HEARTBEAT_TIMEOUT_MS 50

/* 特征缓冲区 - CUDA 固定内存 */
extern float* g_feature_ring_buffer;

/* 缓存行对齐的原子序列计数器 */
alignas(64) extern uint64_t g_sequence_counter;

/* 缓存行对齐的心跳计数器 */
alignas(64) extern uint64_t g_ocaml_heartbeat;
alignas(64) extern uint64_t g_scheme_heartbeat;

/* 缓存行对齐的紧急停止标志 */
alignas(64) extern uint32_t g_emergency_stop;

/* GPU 状态 */
extern int g_cuda_available;
extern int g_cuda_device_id;

/* 核心函数接口 */
int init_cuda_context(void);
void cleanup_cuda_context(void);
void* load_torch_model(const char* path);
float run_inference(void* model_ptr);
void free_torch_model(void* model_ptr);
float* get_feature_buffer_ptr(void);
uint64_t* get_sequence_counter_ptr(void);
void write_features(const float* data, size_t count);
uint64_t read_sequence(void);
void update_ocaml_heartbeat(void);
void update_scheme_heartbeat(void);
int check_system_health(void);
void trigger_emergency_stop(void);
int is_emergency_stopped(void);
void clear_emergency_stop(void);
int get_cuda_device_count(void);
const char* get_cuda_device_name(void);
size_t get_cuda_free_memory(void);

#ifdef __cplusplus
}
#endif

#endif /* LIBQUANT_CORE_H */
