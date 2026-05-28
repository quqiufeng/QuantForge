(* ============================================================
 * QuantForge - C FFI 绑定 (ffi.ml)
 * ============================================================ *)

(* 外部函数声明 - 直接绑定 C 符号 *)
external init_system : unit -> int = "caml_qf_init"
external cleanup_system : unit -> unit = "caml_qf_cleanup"
external get_buffer_ptr : unit -> nativeint = "caml_qf_get_buffer_ptr"
external read_sequence : unit -> int64 = "caml_qf_read_sequence"
external write_features : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t -> int -> unit = "caml_qf_write_features"
external update_heartbeat : unit -> unit = "caml_qf_update_ocaml_heartbeat"
external check_emergency : unit -> int = "caml_qf_is_emergency"
external trigger_emergency : unit -> unit = "caml_qf_emergency_stop"
external check_health : unit -> int = "caml_qf_check_health"
external cuda_available : unit -> int = "caml_qf_cuda_available"

(* 封装函数 *)
let init () =
  let result = init_system () in
  if result <> 0 then
    failwith (Printf.sprintf "qf_init failed with code %d" result)

let cleanup () = cleanup_system ()
let is_emergency () = check_emergency () <> 0
let is_cuda_available () = cuda_available () = 1
