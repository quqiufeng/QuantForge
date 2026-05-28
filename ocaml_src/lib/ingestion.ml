(* ============================================================
 * QuantForge - 高频行情接入与裸指针写入 (ingestion.ml)
 * ============================================================ *)

open Types

(* 全局状态 *)
let feature_buffer : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t option ref = ref None
let heartbeat_counter = ref 0
let heartbeat_interval = 100

(* 外部函数：从 nativeint 创建 Bigarray *)
external bigarray_of_ptr : nativeint -> (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t = "caml_bigarray_of_ptr"

(* 初始化 *)
let init () =
  Ffi.init ();
  let ptr = Ffi.get_buffer_ptr () in
  let ba = bigarray_of_ptr ptr in
  feature_buffer := Some ba;
  Printf.printf "[INFO] Ingestion initialized, CUDA: %b\n" (Ffi.is_cuda_available ())

let cleanup () =
  Ffi.cleanup ()

(* 写入 Tick 到 C 缓冲区 *)
let write_tick_to_c (tick : tick) =
  match !feature_buffer with
  | None -> failwith "Buffer not initialized"
  | Some buffer ->
    (* 直接写入 - 约 1-2ns per set *)
    Bigarray.Array1.set buffer 0 tick.bid;
    Bigarray.Array1.set buffer 1 tick.ask;
    Bigarray.Array1.set buffer 2 tick.bid_qty;
    Bigarray.Array1.set buffer 3 tick.ask_qty;
    Bigarray.Array1.set buffer 4 tick.last_price;
    
    (* 衍生特征 *)
    let spread = tick.ask -. tick.bid in
    let mid_price = (tick.bid +. tick.ask) *. 0.5 in
    let total_qty = tick.bid_qty +. tick.ask_qty in
    let imbalance = 
      if total_qty > 0.0 then (tick.bid_qty -. tick.ask_qty) /. total_qty
      else 0.0
    in
    
    Bigarray.Array1.set buffer 5 spread;
    Bigarray.Array1.set buffer 6 mid_price;
    Bigarray.Array1.set buffer 7 imbalance;
    Bigarray.Array1.set buffer 8 0.0;
    Bigarray.Array1.set buffer 9 0.0;
    Bigarray.Array1.set buffer 10 0.0;
    Bigarray.Array1.set buffer 11 0.0;
    
    (* 调用 C 函数更新序列号 (含内存屏障) *)
    Ffi.write_features buffer 12;
    
    (* 更新心跳 *)
    heartbeat_counter := !heartbeat_counter + 1;
    if !heartbeat_counter mod heartbeat_interval = 0 then
      Ffi.update_heartbeat ()

(* 安全写入 *)
let safe_write_tick tick =
  if Ffi.is_emergency () then
    failwith "Emergency stop triggered!";
  write_tick_to_c tick

(* 状态查询 *)
let get_heartbeat_counter () = !heartbeat_counter
let get_current_sequence () = Ffi.read_sequence ()
