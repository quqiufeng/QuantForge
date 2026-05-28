;;; ============================================================
;;; QuantForge - 高频无锁轮询主控制循环 (main.ss)
;;; ============================================================
;;; 
;;; 设计规范:
;;; - 毫秒级轮询
;;; - 零分配: 禁止 cons/list/vector
;;; - 使用原始数值和符号传递
;;; - 尾递归循环

;; 导入模块
(load "ffi.ss")
(load "strategy.ss")

;;; ============================================================
;;; 全局状态
;;; ============================================================

;; 模型句柄
(define *model-hd* #f)

;; 轮询状态
(define *last-sequence* 0)
(define *poll-counter* 0)
(define *inference-counter* 0)

;; 性能统计
(define *start-time* #f)
(define *last-report-time* #f)

;;; ============================================================
;;; 初始化
;;; ============================================================

(define (qf-init)
  (printf "==========================================\n")
  (printf "QuantForge Chez Scheme Strategy Engine\n")
  (printf "==========================================\n")
  
  ;; 初始化 C 端
  (if (qf-init-system)
      (begin
        (printf "[INFO] System initialized\n")
        (printf "[INFO] CUDA: ~a\n" (= (qf-cuda-available) 1)))
      (begin
        (printf "[FATAL] System initialization failed\n")
        (exit 1)))
  
  ;; 加载模型
  (set! *model-hd* (qf-load-model "c_src/models/quantforge_model_fp16.pt"))
  (if (= *model-hd* 0)
      (begin
        (printf "[WARN] Model not found, running without inference\n")
        (set! *model-hd* #f))
      (printf "[INFO] Model loaded: ~a\n" *model-hd*))
  
  ;; 初始化时间戳
  (set! *start-time* (current-time))
  (set! *last-report-time* *start-time*)
  
  (printf "[INFO] Starting strategy loop...\n")
  (printf "==========================================\n"))

;;; ============================================================
;;; 高频轮询主循环
;;; ============================================================

;; 核心轮询循环 - 零分配版本
;; 严格遵守: 不使用 cons/list/vector
(define (qf-strategy-loop last-seq)
  ;; 1. 读取当前序列号 (Acquire 语义)
  (let ([current-seq (qf-read-sequence)])
    
    ;; 2. 检查是否有新数据
    (if (> current-seq last-seq)
        
        ;; ===== 新数据到达 - 执行策略 =====
        (begin
          ;; 更新计数器
          (set! *inference-counter* (+ *inference-counter* 1))
          
          ;; 3. 更新 Scheme 心跳
          (qf-update-scheme-heartbeat)
          
          ;; 4. 检查系统健康
          (let ([health (qf-check-health)])
            (when (<> health 0)
              (printf "[WARN] Health check failed: ~a\n" health)))
          
          ;; 5. 执行推理 (如果有模型)
          (if *model-hd*
              (begin
                ;; 调用推理
                (let ([err (qf-run-inference-multi
                            *model-hd*
                            (make-ftype-pointer float (foreign-alloc 4))
                            (make-ftype-pointer float (foreign-alloc 4))
                            (make-ftype-pointer float (foreign-alloc 4)))])
                  (if (= err 0)
                      (begin
                        ;; 读取结果
                        (let ([trend (foreign-ref 'float (foreign-alloc 4) 0)]
                              [risk (foreign-ref 'float (foreign-alloc 4) 0)]
                              [vol (foreign-ref 'float (foreign-alloc 4) 0)])
                          
                          ;; 6. 评估仓位
                          (let-values ([(side position risk-level)
                                        (qf-evaluate-position trend risk vol)])
                            
                            ;; 7. 输出信号
                            (when (= (mod *inference-counter* 1000) 0)
                              (printf "[SIGNAL] seq=~a side=~a pos=~a risk=~a trend=~a vol=~a\n"
                                      current-seq side position risk-level trend vol)))))
                      
                      ;; 推理失败
                      (begin
                        (printf "[ERROR] Inference failed: ~a\n" err)
                        (when (>= (mod *inference-counter* 100) 0)
                          (printf "[WARN] Inference failure rate high\n"))))))
              
              ;; 无模型模式 - 只监控
              (begin
                (when (= (mod *inference-counter* 10000) 0)
                  (printf "[MONITOR] seq=~a (no model)\n" current-seq))))
          
          ;; 8. 继续轮询 (更新 last-seq)
          (qf-strategy-loop current-seq))
        
        ;; ===== 无新数据 - 短暂休眠 =====
        (begin
          ;; 计数器递增
          (set! *poll-counter* (+ *poll-counter* 1))
          
          ;; 极短休眠 (让出 CPU)
          ;; 使用 sleep 的纳秒级变体
          ;; (在 Chez 中可用 (sleep (make-time 'time-duration 1000000 0)) 休眠 1ms)
          ;; 但为了不触发 GC，使用更短的方式
          (when (> *poll-counter* 1000)
            (set! *poll-counter* 0)
            ;; 每 1000 次空轮询休眠 1ms
            (sleep (make-time 'time-duration 1000000 0)))
          
          ;; 递归继续
          (qf-strategy-loop last-seq)))))

;;; ============================================================
;;; 性能报告
;;; ============================================================

(define (report-performance)
  (let ([now (current-time)]
        [elapsed (time-difference now *start-time*)])
    (let ([seconds (+ (time-second elapsed)
                      (/ (time-nanosecond elapsed) 1e9))])
      (if (> seconds 0)
          (let ([rate (/ *inference-counter* seconds)])
            (printf "[PERF] Inferences: ~a, Rate: ~a/s, Elapsed: ~as\n"
                    *inference-counter* rate seconds))
          (printf "[PERF] Inferences: ~a\n" *inference-counter*)))))

;;; ============================================================
;;; 清理
;;; ============================================================

(define (qf-cleanup)
  (printf "[INFO] Cleaning up...\n")
  (report-performance)
  (if *model-hd*
      (begin
        (qf-free-model *model-hd*)
        (set! *model-hd* #f)))
  (qf-cleanup)
  (printf "[INFO] Shutdown complete\n"))

;;; ============================================================
;;; 主入口
;;; ============================================================

(define (main)
  ;; 初始化
  (qf-init)
  
  ;; 启动策略循环
  (call/cc
   (lambda (k)
     ;; 信号处理
     (let ([handler (lambda ()
                      (printf "\n[INFO] Received signal, shutting down...\n")
                      (qf-cleanup)
                      (exit 0))])
       
       ;; 捕获中断信号
       ;; 注意: Chez 的信号处理可能因平台而异
       (with-exception-handler
        (lambda (ex)
          (printf "[ERROR] Exception: ~a\n" ex)
          (qf-cleanup)
          (exit 1))
        (lambda ()
          ;; 启动主循环
          (qf-strategy-loop 0)))))))

;; 入口点
(main)
