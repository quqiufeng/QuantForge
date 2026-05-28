/*
 * QuantForge - OCaml C 桥接层
 */

#include <caml/mlvalues.h>
#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/bigarray.h>
#include "libquant_core_c.h"

CAMLprim value caml_qf_init(value unit) {
    return Val_int(qf_init());
}

CAMLprim value caml_qf_cleanup(value unit) {
    qf_cleanup();
    return Val_unit;
}

CAMLprim value caml_qf_get_buffer_ptr(value unit) {
    return caml_copy_nativeint(qf_get_buffer_ptr());
}

CAMLprim value caml_qf_read_sequence(value unit) {
    return caml_copy_int64(qf_read_sequence());
}

CAMLprim value caml_qf_update_ocaml_heartbeat(value unit) {
    qf_update_ocaml_heartbeat();
    return Val_unit;
}

CAMLprim value caml_qf_is_emergency(value unit) {
    return Val_int(qf_is_emergency());
}

CAMLprim value caml_qf_check_health(value unit) {
    return Val_int(qf_check_health());
}

CAMLprim value caml_qf_emergency_stop(value unit) {
    qf_emergency_stop();
    return Val_unit;
}

CAMLprim value caml_qf_cuda_available(value unit) {
    return Val_int(qf_cuda_available());
}

CAMLprim value caml_qf_write_features(value ba, value count_val) {
    float* data = (float*)Caml_ba_data_val(ba);
    size_t count = Int_val(count_val);
    qf_write_features(data, count);
    return Val_unit;
}

/* 创建 Bigarray 指向 C 内存 */
CAMLprim value caml_bigarray_of_ptr(value ptr_val) {
    void* ptr = (void*)Nativeint_val(ptr_val);
    return caml_ba_alloc_dims(
        CAML_BA_FLOAT32 | CAML_BA_C_LAYOUT,
        1,
        ptr,
        1024
    );
}
