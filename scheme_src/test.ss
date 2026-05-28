;;; 测试脚本
(load "strategy.ss")

(display "Testing Alpha Factors:\n")
(display "Momentum: ")
(display ((alpha-factor-calculate-fn momentum-factor) '(100 105 103)))
(newline)

(display "Composite Alpha: ")
(display (get-alpha-signal '(100 105 103 108)))
(newline)
