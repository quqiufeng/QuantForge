(* QuantForge - 行情接入模块 *)

let parse_tick_packet _data =
  let now = Int64.of_float (Unix.gettimeofday () *. 1_000_000.0) in
  Types.create_tick
    ~symbol:"BTC/USD"
    ~bid:50000.0
    ~ask:50001.0
    ~bid_size:1.0
    ~ask_size:1.0
    ~timestamp:now
    ~sequence:1L

let extract_features (tick : Types.tick) =
  [| tick.bid; tick.ask; tick.ask -. tick.bid |]

let validate_tick (tick : Types.tick) =
  tick.bid > 0.0 && tick.ask >= tick.bid
