(lang dune 3.0)
(name quantforge)

(package
 (name quantforge)
 (synopsis "QuantForge - Ultra-low latency AI quantitative trading system")
 (description "OCaml 4 ingestion engine with Lwt async event loop")
 (depends
  (lwt (>= 5.6.0))
  (lwt_ppx (>= 2.0.0))))
