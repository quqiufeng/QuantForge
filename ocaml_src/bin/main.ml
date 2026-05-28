(* ============================================================
 * QuantForge - Lwt 异步行情模拟器主入口 (main.ml)
 * ============================================================ *)

open Lwt
open Quantforge

(* 全局配置 *)
let log_interval = 10000
let tick_counter = ref 0L
let gen_counter = ref 0L
let base_price = ref 50000.0

(* 模拟 Tick 生成 *)
let generate_tick () =
  gen_counter := Int64.add !gen_counter 1L;
  let now_us = Int64.of_float (Unix.gettimeofday () *. 1_000_000.0) in
  let noise = (Random.float 2.0 -. 1.0) *. 10.0 in
  base_price := !base_price +. noise *. 0.01;
  let bid = !base_price -. 5.0 in
  let ask = !base_price +. 5.0 in
  {
    Types.symbol = "BTC/USDT";
    bid;
    ask;
    bid_qty = 1.0 +. Random.float 5.0;
    ask_qty = 1.0 +. Random.float 5.0;
    last_price = !base_price;
    last_qty = 0.1;
    volume_24h = 1000000.0;
    timestamp = now_us;
    sequence = !gen_counter;
  }

(* 主循环 - 零分配版本 *)
let rec ingestion_loop () =
  if Ffi.is_emergency () then begin
    Lwt_io.printl "[FATAL] Emergency stop detected!"
    >>= fun () -> Lwt.fail (Failure "Emergency stop")
  end
  else begin
    let tick = generate_tick () in
    Ingestion.write_tick_to_c tick;
    tick_counter := Int64.add !tick_counter 1L;
    
    let count = !tick_counter in
    if Int64.rem count (Int64.of_int log_interval) = 0L then begin
      let seq = Ingestion.get_current_sequence () in
      Lwt_io.printf "[TICK] Count: %Ld, Seq: %Ld, Bid: %.2f\n" count seq tick.bid
      >>= fun () -> ingestion_loop ()
    end
    else
      ingestion_loop ()
  end

(* 心跳监控 *)
let rec heartbeat_loop () =
  Lwt_unix.sleep 0.1
  >>= fun () ->
  let _ = Ffi.check_health () in
  heartbeat_loop ()

(* 序列号监控 *)
let rec sequence_monitor_loop () =
  Lwt_unix.sleep 1.0
  >>= fun () ->
  let seq = Ffi.read_sequence () in
  let count = !tick_counter in
  Lwt_io.printf "[STATS] Ticks: %Ld, Sequence: %Ld\n" count seq
  >>= fun () -> sequence_monitor_loop ()

(* 主入口 *)
let main () =
  Lwt_io.printl "=========================================="
  >>= fun () ->
  Lwt_io.printl "QuantForge OCaml 4 Ingestion Engine"
  >>= fun () ->
  Lwt_io.printl "=========================================="
  >>= fun () ->
  Random.self_init ();
  (try Ingestion.init ()
   with e -> 
     Printf.eprintf "[ERROR] Init failed: %s\n" (Printexc.to_string e);
     exit 1);
  Lwt_io.printl "[INFO] Starting ingestion loop..."
  >>= fun () ->
  Lwt.pick [
    ingestion_loop ();
    heartbeat_loop ();
    sequence_monitor_loop ();
  ]

(* 信号处理 *)
let () =
  let handler _ =
    Printf.printf "\n[INFO] Shutting down...\n";
    Ingestion.cleanup ();
    exit 0
  in
  Sys.set_signal Sys.sigint (Sys.Signal_handle handler);
  Sys.set_signal Sys.sigterm (Sys.Signal_handle handler);
  
  try Lwt_main.run (main ())
  with e ->
    Printf.eprintf "[FATAL] %s\n" (Printexc.to_string e);
    Ingestion.cleanup ();
    exit 1
