;;; ============================================================
;;; QuantForge - Chez Scheme FFI 接口映射 (ffi.ss)
;;; ============================================================
;;; 
;;; 设计规范:
;;; - 使用 (load-shared-object) 载入动态库
;;; - C 侧 intptr_t → Chez 侧 iptr
;;; - C 侧 uint64_t → Chez 侧 unsigned-64
;;; - 所有指针类型严格映射为 iptr

;;; 载入 C++ 动态库
(load-shared-object "./c_src/libquant_core.so")

;;; ============================================================
;;; 系统初始化
;;; ============================================================

(define qf-init
  (foreign-procedure "qf_init" () int))

(define qf-cleanup
  (foreign-procedure "qf_cleanup" () void))

;;; ============================================================
;;; 缓冲区操作
;;; ============================================================

;; 获取特征缓冲区指针 (返回 iptr)
(define qf-get-buffer-ptr
  (foreign-procedure "qf_get_buffer_ptr" () iptr))

;; 读取序列号 (返回 unsigned-64)
(define qf-read-sequence
  (foreign-procedure "qf_read_sequence" () unsigned-64))

;; 写入特征数据
;; 参数: data-ptr (iptr), count (int)
(define qf-write-features
  (foreign-procedure "qf_write_features" (iptr int) void))

;;; ============================================================
;;; 模型管理
;;; ============================================================

;; 加载模型
;; 参数: path (string)
;; 返回: iptr (模型句柄)
(define qf-load-model
  (foreign-procedure "qf_load_model" (string) iptr))

;; 释放模型
;; 参数: model (iptr)
(define qf-free-model
  (foreign-procedure "qf_free_model" (iptr) void))

;; 检查模型有效性
;; 参数: model (iptr)
;; 返回: int (0=有效)
(define qf-is-model-valid
  (foreign-procedure "qf_is_model_valid" (iptr) int))

;;; ============================================================
;;; 推理接口
;;; ============================================================

;; 单输出推理 (返回风险概率)
;; 参数: model (iptr)
;; 返回: float
(define qf-run-inference
  (foreign-procedure "qf_run_inference" (iptr) float))

;; 多输出推理 (推荐)
;; 参数: model (iptr), out-trend (* float), out-risk (* float), out-vol (* float)
;; 返回: int (0=成功)
(define qf-run-inference-multi
  (foreign-procedure "qf_run_inference_multi"
    (iptr (* float) (* float) (* float))
    int))

;;; ============================================================
;;; 心跳与断路器
;;; ============================================================

;; 更新 OCaml 心跳
(define qf-update-ocaml-heartbeat
  (foreign-procedure "qf_update_ocaml_heartbeat" () void))

;; 更新 Scheme 心跳
(define qf-update-scheme-heartbeat
  (foreign-procedure "qf_update_scheme_heartbeat" () void))

;; 检查系统健康
;; 返回: int (0=健康, 1=OCaml超时, 2=Scheme超时)
(define qf-check-health
  (foreign-procedure "qf_check_health" () int))

;; 获取 OCaml 心跳
(define qf-get-ocaml-heartbeat
  (foreign-procedure "qf_get_ocaml_heartbeat" () unsigned-64))

;; 获取 Scheme 心跳
(define qf-get-scheme-heartbeat
  (foreign-procedure "qf_get_scheme_heartbeat" () unsigned-64))

;; 触发紧急停止
(define qf-emergency-stop
  (foreign-procedure "qf_emergency_stop" () void))

;; 检查是否紧急停止
;; 返回: int (1=已触发)
(define qf-is-emergency
  (foreign-procedure "qf_is_emergency" () int))

;; 清除紧急停止
(define qf-clear-emergency
  (foreign-procedure "qf_clear_emergency" () void))

;;; ============================================================
;;; GPU 信息
;;; ============================================================

;; 检查 CUDA 可用性
;; 返回: int (1=可用)
(define qf-cuda-available
  (foreign-procedure "qf_cuda_available" () int))

;; 获取 CUDA 设备数量
(define qf-cuda-device-count
  (foreign-procedure "qf_cuda_device_count" () int))

;; 获取 CUDA 可用显存 (字节)
(define qf-cuda-free-memory
  (foreign-procedure "qf_cuda_free_memory" () unsigned-64))

;;; ============================================================
;;; 高级封装
;;; ============================================================

;; 初始化系统并返回状态
(define (qf-init-system)
  (let ([result (qf-init)])
    (if (= result 0)
        (begin
          (printf "[INFO] QuantForge Scheme FFI initialized\n")
          (printf "[INFO] CUDA available: ~a\n" (= (qf-cuda-available) 1))
          #t)
        (begin
          (printf "[ERROR] qf_init failed with code ~a\n" result)
          #f))))

;; 检查紧急停止 (返回 boolean)
(define (qf-check-emergency?)
  (not (= (qf-is-emergency) 0)))

;; 安全推理 - 自动检查紧急停止
(define (qf-safe-inference model)
  (if (qf-check-emergency?)
      (begin
        (printf "[WARN] Emergency stop active, skipping inference\n")
        #f)
      (begin
        (qf-run-inference model)
        #t)))

;; 获取推理结果到 Scheme 变量
;; 使用预分配的 float 数组接收结果
(define qf-inference-results
  ;; 预分配三个 float 的静态数组
  (let ([trend-buf (make-ftype-pointer float (foreign-alloc (* 4 3)))])
    (lambda (model)
      (let ([risk-buf (ftype-pointer-address trend-buf)]
            [trend-ptr trend-buf]
            [risk-ptr (make-ftype-pointer float (+ risk-buf 4))]
            [vol-ptr (make-ftype-pointer float (+ risk-buf 8))])
        (let ([err (qf-run-inference-multi model trend-ptr risk-ptr vol-ptr)])
          (if (= err 0)
              (values (ftype-ref float () trend-ptr)
                      (ftype-ref float () risk-ptr)
                      (ftype-ref float () vol-ptr))
              (begin
                (printf "[ERROR] Inference failed: ~a\n" err)
                (values 0.0 1.0 0.0))))))))  ; 保守默认值
