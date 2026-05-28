(* ============================================================
 * QuantForge - 类型定义 (types.ml)
 * ============================================================ *)

(* 交易方向 *)
type side = Buy | Sell

(* 订单状态 *)
type order_status = Pending | PartialFilled | Filled | Cancelled | Rejected

(* 订单类型 *)
type order_type = Limit | Market | StopLoss | TakeProfit

(* 订单记录 *)
type order = {
  id : int64;
  symbol : string;
  side : side;
  order_type : order_type;
  price : float;
  quantity : float;
  filled_qty : float;
  status : order_status;
  timestamp : int64;
  update_time : int64;
}

(* 价格层级 *)
type price_level = {
  price : float;
  quantity : float;
  order_count : int;
}

(* 订单簿 *)
type orderbook = {
  symbol : string;
  bids : price_level list;
  asks : price_level list;
  timestamp : int64;
  sequence : int64;
}

(* Tick 行情 *)
type tick = {
  symbol : string;
  bid : float;
  ask : float;
  bid_qty : float;
  ask_qty : float;
  last_price : float;
  last_qty : float;
  volume_24h : float;
  timestamp : int64;
  sequence : int64;
}

(* 持仓 *)
type position = {
  symbol : string;
  side : side;
  quantity : float;
  entry_price : float;
  mark_price : float;
  unrealized_pnl : float;
}

(* 风控配置 *)
type risk_config = {
  max_drawdown : float;
  max_position_size : float;
  max_order_size : float;
  max_orders_per_sec : int;
  price_deviation_pct : float;
  daily_loss_limit : float;
}

(* 风控状态 *)
type risk_state = Normal | Warning | Halted of string

(* 风控引擎 *)
type risk_engine = {
  config : risk_config;
  mutable state : risk_state;
  mutable daily_pnl : float;
  mutable order_count : int;
  mutable last_order_time : int64;
}

(* 默认风控配置 *)
let default_risk_config = {
  max_drawdown = 0.05;
  max_position_size = 1.0;
  max_order_size = 0.1;
  max_orders_per_sec = 100;
  price_deviation_pct = 0.01;
  daily_loss_limit = 0.02;
}

(* 创建风控引擎 *)
let create_risk_engine config = {
  config;
  state = Normal;
  daily_pnl = 0.0;
  order_count = 0;
  last_order_time = 0L;
}

(* 检查订单风控 *)
let check_order_risk engine (ord : order) current_price =
  match engine.state with
  | Halted reason -> Error ("System halted: " ^ reason)
  | _ ->
    if ord.quantity > engine.config.max_order_size then
      Error "Order size exceeds limit"
    else if abs_float (ord.price -. current_price) /. current_price 
            > engine.config.price_deviation_pct then
      Error "Price deviation too large"
    else if engine.daily_pnl < -.engine.config.daily_loss_limit then begin
      engine.state <- Halted "Daily loss limit reached";
      Error "Daily loss limit reached"
    end
    else if engine.order_count >= engine.config.max_orders_per_sec then
      Error "Order rate limit exceeded"
    else
      Ok ()
