/*
 * libquant_core_c.h - C 兼容头文件
 */

#ifndef LIBQUANT_CORE_C_H
#define LIBQUANT_CORE_C_H

#include <stdint.h>
#include <stddef.h>

/* 类型定义 */
typedef intptr_t  qf_model_t;
typedef intptr_t  qf_buffer_t;
typedef uint64_t  qf_sequence_t;
typedef int32_t   qf_error_t;

/* 错误码 */
#define QF_OK                    0
#define QF_ERROR_NULL_PTR       -1
#define QF_ERROR_INFERENCE      -999
#define QF_FEATURE_BUFFER_SIZE  1024

/* C 函数声明 */
#ifdef __cplusplus
extern "C" {
#endif

qf_error_t qf_init(void);
void qf_cleanup(void);
qf_buffer_t qf_get_buffer_ptr(void);
qf_sequence_t qf_read_sequence(void);
void qf_write_features(const float* data, size_t count);
void qf_update_ocaml_heartbeat(void);
int32_t qf_is_emergency(void);
int32_t qf_check_health(void);
void qf_emergency_stop(void);
int32_t qf_cuda_available(void);

#ifdef __cplusplus
}
#endif

#endif
