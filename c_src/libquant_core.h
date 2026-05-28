/*
 * ============================================================
 * QuantForge - 超低延迟量化交易系统
 * C ABI 接口定义 (libquant_core.h)
 * ============================================================
 * 
 * 设计规范:
 * 1. extern "C" 确保符号不被 C++ 名称修饰
 * 2. qf_ 前缀统一命名空间
 * 3. intptr_t 替代 void* 确保 32/64 位兼容
 * 4. alignas(64) 缓存行对齐防止 False Sharing
 * 5. 所有内存屏障在 C++ 实现层完成
 */

#ifndef LIBQUANT_CORE_H
#define LIBQUANT_CORE_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * 类型定义
 * ============================================================ */

typedef intptr_t  qf_model_t;
typedef intptr_t  qf_buffer_t;
typedef uint64_t  qf_sequence_t;
typedef int32_t   qf_error_t;

#define QF_OK                    0
#define QF_ERROR_NULL_PTR       -1
#define QF_ERROR_MODEL_LOAD     -2
#define QF_ERROR_MODEL_INVALID  -3
#define QF_ERROR_EMERGENCY      -4
#define QF_ERROR_CUDA_DEVICE    -5
#define QF_ERROR_INFERENCE      -999

#define QF_FEATURE_BUFFER_SIZE  1024
#define QF_HEARTBEAT_TIMEOUT_MS 20

/* ============================================================
 * 全局状态变量声明
 * ============================================================ */

extern float* g_qf_feature_buffer;

alignas(64) extern uint64_t g_qf_sequence_counter;
alignas(64) extern uint64_t g_qf_ocaml_heartbeat;
alignas(64) extern uint64_t g_qf_scheme_heartbeat;
alignas(64) extern uint32_t g_qf_emergency_stop;

extern int32_t g_qf_cuda_available;
extern int32_t g_qf_cuda_device_id;

/* ============================================================
 * 初始化与清理
 * ============================================================ */

qf_error_t qf_init(void);
void qf_cleanup(void);

/* ============================================================
 * 缓冲区操作
 * ============================================================ */

qf_buffer_t qf_get_buffer_ptr(void);
qf_buffer_t qf_get_sequence_ptr(void);
qf_sequence_t qf_read_sequence(void);
void qf_write_features(const float* data, size_t count);
qf_error_t qf_read_features_safe(float* output, size_t count, qf_sequence_t* out_seq);

/* ============================================================
 * 模型管理
 * ============================================================ */

qf_model_t qf_load_model(const char* path);
void qf_free_model(qf_model_t model);
qf_error_t qf_is_model_valid(qf_model_t model);

/* ============================================================
 * 推理接口
 * ============================================================ */

float qf_run_inference(qf_model_t model);
qf_error_t qf_run_inference_multi(qf_model_t model, float* out_trend, float* out_risk, float* out_vol);

/* ============================================================
 * 心跳与健康检查
 * ============================================================ */

void qf_update_ocaml_heartbeat(void);
void qf_update_scheme_heartbeat(void);
int32_t qf_check_health(void);
uint64_t qf_get_ocaml_heartbeat(void);
uint64_t qf_get_scheme_heartbeat(void);

/* ============================================================
 * 紧急停止控制
 * ============================================================ */

void qf_emergency_stop(void);
int32_t qf_is_emergency(void);
void qf_clear_emergency(void);

/* ============================================================
 * GPU 信息
 * ============================================================ */

int32_t qf_cuda_available(void);
int32_t qf_cuda_device_count(void);
const char* qf_cuda_device_name(void);
uint64_t qf_cuda_free_memory(void);
void qf_prefetch_buffer(void);

#ifdef __cplusplus
}
#endif

#endif /* LIBQUANT_CORE_H */
