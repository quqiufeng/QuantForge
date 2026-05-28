;;; ============================================================
;;; QuantForge - 动态策略与仓位控制内核 (strategy.ss)
;;; ============================================================
;;; 
;;; 设计规范:
;;; - 顶层状态变量保持持久性
;;; - 零分配: 避免 cons/list/vector
;;; - 凯利公式动态仓位
;;; - 智能网格套利

;;; ============================================================
;;; 核心状态变量 (全局持久化)
;;; ============================================================

;; 基础配置
(define *base-capital* 100000.0)      ;; 总资本
(define *beta-pool-ratio* 0.6)        ;; Beta 定投池比例 (60%)
(define *alpha-pool-ratio* 0.4)       ;; Alpha 套利池比例 (40%)

;; 资金池
(define *beta-pool-nav* (* *base-capital* *beta-pool-ratio*))
(define *alpha-pool-nav* (* *base-capital* *alpha-pool-ratio*))

;; 当前持仓
(define *current-position* 0.0)       ;; 当前仓位比例
(define *current-side* 'hold)         ;; 当前方向: buy/sell/hold

;; 风控状态
(define *risk-lock* #f)               ;; 风险锁死标志
(define *max-risk-threshold* 0.8)     ;; 最大风险阈值
(define *emergency-pause-ms* 5000)    ;; 紧急暂停时间 (ms)

;; 定投状态
(define *last-dca-time* 0)            ;; 上次定投时间戳
(define *dca-interval-ms* 3600000)    ;; 定投间隔 (1小时)
(define *dca-base-amount* 100.0)      ;; 基础定投金额
(define *dca-max-amount* 1000.0)      ;; 最大定投金额

;; 网格状态
(define *grid-center* 50000.0)        ;; 网格中心价格
(define *grid-step* 100.0)            ;; 网格步长
(define *grid-levels* 5)              ;; 网格层数
(define *grid-orders* #f)             ;; 当前网格订单

;; 凯利公式参数
(define *kelly-half* #t)              ;; 使用半凯利 (更保守)
(define *max-position-limit* 0.6)     ;; 最大仓位限制

;; Alpha 因子热插拔钩子
;; 默认识别函数: 直接返回原始信号
(define alpha-factor-hook
  (lambda (trend risk vol)
    ;; 默认: 直接透传
    (values trend risk vol)))

;; 更新钩子 (运行时热更新)
(define (set-alpha-factor-hook! new-hook)
  (set! alpha-factor-hook new-hook)
  (printf "[INFO] Alpha factor hook updated\n"))

;;; ============================================================
;;; 风控与断路器
;;; ============================================================

;; 检查风险断路
(define (check-risk-circuit-breaker risk)
  (cond
    ;; 极端风险: 立即锁死
    [(> risk *max-risk-threshold*)
     (set! *risk-lock* #t)
     (set! *current-side* 'hold)
     (printf "[ALERT] Risk circuit breaker triggered! Risk=~a\n" risk)
     #f]  ;; 拒绝交易
    ;; 高风险: 警告但不锁死
    [(> risk 0.6)
     (printf "[WARN] High risk detected: ~a, reducing exposure\n" risk)
     #t]  ;; 允许但减小仓位
    ;; 正常
    [else #t]))

;; 清除风险锁 (手动)
(define (clear-risk-lock!)
  (set! *risk-lock* #f)
  (printf "[INFO] Risk lock cleared\n"))

;;; ============================================================
;;; 动态定投计算
;;; ============================================================

;; 计算动态定投金额
;; 零分配: 只使用数值运算
(define (calculate-dca-amount trend risk volatility current-time)
  ;; 检查风险锁
  (if *risk-lock*
      0.0  ;; 锁死时不定投
      (let ([base *dca-base-amount*]
            [max-amt *dca-max-amount*])
        (cond
          ;; 风险过高: 停止定投
          [(> risk *max-risk-threshold*) 0.0]
          
          ;; 趋势为负: 减少投入
          [(< trend 0.0)
           (* base 0.5 (- 1.0 risk))]
          
          ;; 趋势为正: 根据信号强度和风险调整
          [else
           (let ([trend-weight (min 1.0 (+ 0.5 (* 0.5 trend)))]
                 [risk-weight (- 1.0 risk)]
                 [vol-weight (/ 1.0 (+ 1.0 (* 2.0 volatility)))])
             (min (* base trend-weight risk-weight vol-weight)
                  max-amt))]))))

;;; ============================================================
;;; 凯利公式动态仓位
;;; ============================================================

;; 凯利公式计算最优仓位
;; 零分配: 纯数值运算
(define (kelly-criterion expected-return volatility max-position)
  (let* ([variance (* volatility volatility)]
         [epsilon 1e-6]
         [kelly-raw (/ expected-return (+ variance epsilon))]
         [kelly-value (if *kelly-half*
                         (* 0.5 kelly-raw)  ;; 半凯利
                         kelly-raw)]
         [vol-scaling (/ 1.0 (+ 1.0 (* 5.0 volatility)))]
         [f-star (* kelly-value vol-scaling)])
    (max 0.0 (min f-star max-position))))

;; 根据模型输出计算目标仓位
(define (calculate-target-position trend risk volatility)
  (if *risk-lock*
      0.0  ;; 锁死时零仓位
      (let ([kelly-pos (kelly-criterion trend volatility *max-position-limit*)])
        ;; 根据风险进一步缩放
        (* kelly-pos (- 1.0 risk)))))

;;; ============================================================
;;; 智能网格套利
;;; ============================================================

;; 计算动态网格步长
(define (calculate-grid-step current-price volatility holding-period-sec)
  (let* ([time-scaling (sqrt (/ holding-period-sec 3600.0))]
         [k 0.6]
         [step (* current-price volatility time-scaling k)])
    (max step 1.0)))  ;; 最小步长 1.0

;; 计算网格层数
(define (calculate-grid-levels capital risk-per-grid)
  (let ([max-levels 10])
    (min max-levels
         (max 3
              (floor (/ capital risk-per-grid))))))

;; 生成网格订单价格
;; 返回 buy-price 和 sell-price (不使用 list!)
(define (generate-grid-prices center step levels)
  (values (- center step)
          (+ center step)))

;;; ============================================================
;;; 策略评估主函数
;;; ============================================================

;; 评估仓位状态 (核心入口)
;; 零分配: 不使用 cons/list/vector
(define (qf-evaluate-position trend risk volatility)
  ;; 1. 应用 Alpha 因子钩子 (热插拔)
  (let-values ([(adj-trend adj-risk adj-vol) (alpha-factor-hook trend risk volatility)])
    
    ;; 2. 检查风险断路器
    (if (not (check-risk-circuit-breaker adj-risk))
        (begin
          ;; 紧急状态: 强制平仓
          (set! *current-position* 0.0)
          (set! *current-side* 'hold)
          (printf "[EMERGENCY] Position cleared due to high risk\n")
          (values 'hold 0.0 0.0))
        
        ;; 3. 计算目标仓位
        (let ([target-pos (calculate-target-position adj-trend adj-risk adj-vol)])
          
          ;; 4. 更新当前仓位
          (set! *current-position* target-pos)
          
          ;; 5. 确定交易方向
          (cond
            [(> target-pos 0.3)
             (set! *current-side* 'buy)
             (printf "[SIGNAL] BUY position=~a trend=~a risk=~a\n"
                     target-pos adj-trend adj-risk)
             (values 'buy target-pos adj-risk)]
            
            [(< target-pos -0.3)
             (set! *current-side* 'sell)
             (printf "[SIGNAL] SELL position=~a trend=~a risk=~a\n"
                     target-pos adj-trend adj-risk)
             (values 'sell (abs target-pos) adj-risk)]
            
            [else
             (set! *current-side* 'hold)
             (values 'hold target-pos adj-risk)])))))

;;; ============================================================
;;; 仓位重平衡
;;; ============================================================

;; 执行仓位调整
(define (rebalance-position target-position current-price)
  (let* ([nav *alpha-pool-nav*]
         [target-value (* nav target-position)]
         [current-value (* nav *current-position*)]
         [delta (- target-value current-value)]
         [delta-qty (/ delta current-price)])
    
    (if (> (abs delta-qty) 0.001)  ;; 最小调整阈值
        (begin
          (if (> delta 0)
              (printf "[REBALANCE] BUY ~a @ ~a\n" delta-qty current-price)
              (printf "[REBALANCE] SELL ~a @ ~a\n" (abs delta-qty) current-price))
          #t)
        #f)))

;;; ============================================================
;;; 网格套利执行
;;; ============================================================

;; 更新网格
(define (update-grid current-price volatility)
  (let ([step (calculate-grid-step current-price volatility 3600.0)])
    (set! *grid-step* step)
    (set! *grid-center* current-price)
    (printf "[GRID] Updated center=~a step=~a\n" current-price step)))

;;; ============================================================
;;; 状态查询
;;; ============================================================

(define (get-current-position) *current-position*)
(define (get-current-side) *current-side*)
(define (get-beta-nav) *beta-pool-nav*)
(define (get-alpha-nav) *alpha-pool-nav*)
(define (is-risk-locked?) *risk-lock*)

;;; ============================================================
;;; 示例 Alpha 因子 (可热插拔)
;;; ============================================================

;; 动量因子
(define (momentum-factor prices)
  (if (< (length prices) 2)
      0.0
      (/ (- (car prices) (cadr prices))
         (cadr prices))))

;; 均值回归因子
(define (mean-reversion-factor prices)
  (if (null? prices)
      0.0
      (let* ([n (length prices)]
             [mean (/ (apply + prices) n)]
             [current (car prices)])
        (/ (- mean current) mean))))

;; 波动率因子
(define (volatility-factor prices)
  (if (< (length prices) 2)
      0.0
      (let* ([n (length prices)]
             [mean (/ (apply + prices) n)]
             [variance (/ (apply + (map (lambda (x) (expt (- x mean) 2)) prices))
                         n)])
        (sqrt variance))))
