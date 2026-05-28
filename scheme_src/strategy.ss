;;; QuantForge - Alpha 因子定义
;;; 动态可热插拔的策略模块

;;; Alpha 因子基类
(define-record-type alpha-factor
  (fields
    name           ; 因子名称
    weight         ; 权重
    calculate-fn   ; 计算函数
    enabled?))     ; 是否启用

;;; 创建 Alpha 因子
(define (make-factor name weight calculate-fn)
  (make-alpha-factor name weight calculate-fn #t))

;;; 动量因子
(define momentum-factor
  (make-factor "momentum" 0.3
    (lambda (prices)
      (if (< (length prices) 2)
          0.0
          (let ([current (car prices)]
                [previous (cadr prices)])
            (/ (- current previous) previous))))))

;;; 均值回归因子
(define mean-reversion-factor
  (make-factor "mean-reversion" 0.25
    (lambda (prices)
      (if (null? prices)
          0.0
          (let ([mean (/ (apply + prices) (length prices))]
                [current (car prices)])
            (/ (- mean current) mean))))))

;;; 波动率因子
(define volatility-factor
  (make-factor "volatility" 0.2
    (lambda (prices)
      (if (< (length prices) 2)
          0.0
          (let* ([n (length prices)]
                 [mean (/ (apply + prices) n)]
                 [variance (/ (apply + (map (lambda (x) 
                                              (expt (- x mean) 2)) 
                                            prices)) 
                              n)])
            (sqrt variance))))))

;;; 综合 Alpha 信号计算
(define (calculate-composite-alpha factors prices)
  (let ([enabled-factors (filter alpha-factor-enabled? factors)])
    (if (null? enabled-factors)
        0.5
        (let ([weighted-sum
               (apply + (map (lambda (f)
                               (* (alpha-factor-weight f)
                                  ((alpha-factor-calculate-fn f) prices)))
                             enabled-factors))]
              [total-weight
               (apply + (map alpha-factor-weight enabled-factors))])
          (/ weighted-sum total-weight)))))

;;; 因子注册表
(define *factor-registry*
  (list momentum-factor
        mean-reversion-factor
        volatility-factor))

;;; 注册新因子
(define (register-factor! factor)
  (set! *factor-registry* (cons factor *factor-registry*)))

;;; 注销因子
(define (unregister-factor! name)
  (set! *factor-registry*
        (filter (lambda (f) (not (string=? (alpha-factor-name f) name)))
                *factor-registry*)))

;;; 启用/禁用因子
(define (toggle-factor! name enabled?)
  (set! *factor-registry*
        (map (lambda (f)
               (if (string=? (alpha-factor-name f) name)
                   (make-alpha-factor (alpha-factor-name f)
                                      (alpha-factor-weight f)
                                      (alpha-factor-calculate-fn f)
                                      enabled?)
                   f))
             *factor-registry*)))

;;; 导出接口
(define (get-alpha-signal price-history)
  (calculate-composite-alpha *factor-registry* price-history))
