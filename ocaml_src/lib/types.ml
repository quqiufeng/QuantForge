(* QuantForge - 订单簿与风控类型定义 *)

type side = Buy | Sell
type order_state = Pending | Filled | Cancelled | Rejected
type order_id = int64
type price = float
type quantity = float

type order = {
  id : order_id;
  side : side;
  price : price;
  quantity : quantity;
  timestamp : int64;
  state : order_state;
}

type price_level = {
  price : price;
  total_quantity : quantity;
  order_count : int;
}

type orderbook = {
  symbol : string;
  bids : price_level list;
  asks : price_level list;
  last_update : int64;
  sequence : int64;
}

type tick = {
  symbol : string;
  bid : price;
  ask : price;
  bid_size : quantity;
  ask_size : quantity;
  timestamp : int64;
  sequence : int64;
}

type risk_config = {
  max_position : quantity;
  max_order_size : quantity;
  price_limit_pct : float;
  max_orders_per_second : int;
}

type risk_check_result = Approved | Rejected of string

let create_tick ~symbol ~bid ~ask ~bid_size ~ask_size ~timestamp ~sequence =
  { symbol; bid; ask; bid_size; ask_size; timestamp; sequence }

let create_order ~id ~side ~price ~quantity ~timestamp =
  { id; side; price; quantity; timestamp; state = Pending }

let best_bid = function
  | [] -> None
  | { price; _ } :: _ -> Some price

let best_ask = function
  | [] -> None
  | { price; _ } :: _ -> Some price
