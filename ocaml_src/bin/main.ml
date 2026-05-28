(* QuantForge - Lwt 异步行情接入主循环 *)

open Lwt
open Quantforge

let buffer_size = 1024

let simulate_tick_stream () =
  let counter = ref 0L in
  fun () ->
    counter := Int64.add !counter 1L;
    let now = Int64.of_float (Unix.gettimeofday () *. 1_000_000.0) in
    Lwt.return (Types.create_tick
      ~symbol:"BTC/USD"
      ~bid:(50000.0 +. Random.float 100.0)
      ~ask:(50001.0 +. Random.float 100.0)
      ~bid_size:(Random.float 10.0)
      ~ask_size:(Random.float 10.0)
      ~timestamp:now
      ~sequence:!counter)

let write_tick_to_buffer buffer (tick : Types.tick) =
  Bigarray.Array1.set buffer 0 tick.bid;
  Bigarray.Array1.set buffer 1 tick.ask;
  Lwt.return_unit

let rec ingestion_loop tick_source buffer tick_count =
  let%lwt tick = tick_source () in
  let%lwt () = write_tick_to_buffer buffer tick in
  let new_count = Int64.add tick_count 1L in
  if Int64.rem new_count 1000L = 0L then
    Lwt_io.printf "Processed %Ld ticks\n" new_count
    >>= fun () -> ingestion_loop tick_source buffer new_count
  else
    ingestion_loop tick_source buffer new_count

let rec heartbeat_loop () =
  let%lwt () = Lwt_unix.sleep 1.0 in
  Lwt_io.printf "[Heartbeat]\n"
  >>= fun () -> heartbeat_loop ()

let main () =
  Lwt_io.printf "QuantForge Starting...\n"
  >>= fun () ->
  Random.self_init ();
  let tick_source = simulate_tick_stream () in
  let buffer = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout buffer_size in
  Lwt.pick [
    ingestion_loop tick_source buffer 0L;
    heartbeat_loop ();
  ]

let () = Lwt_main.run (main ())
