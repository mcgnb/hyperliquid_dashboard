from typing import List, Union
import ccxt.async_support as ccxt
import asyncio
import pandas as pd
from datetime import datetime
from pydantic import BaseModel
from decimal import Decimal, getcontext

# ---------------------
# Pydantic Models
# ---------------------
class UsdcBalance(BaseModel):
    total: float
    free: float
    used: float

class Info(BaseModel):
    success: bool
    message: str

# ---------------------
# Market helper
# ---------------------
class Market(BaseModel):
    internal_pair: str
    base: str
    quote: str
    price_precision: float
    contract_precision: float
    contract_size: float = 1.0
    min_contracts: float = 1.0
    max_contracts: float = float('inf')
    min_cost: float = 0.0
    max_cost: float = float('inf')
    coin_index: int = 0
    market_price: float = 0.0

# ---------------------
# Main Hyperliquid Class
# ---------------------
class PerpHyperliquid:
    def __init__(self, public_api=None):
        self.public_address = public_api
        # No secret key is provided since public endpoints are used.
        self._session = ccxt.hyperliquid()
        self.market: dict[str, Market] = {}

    async def load_markets(self):
        data = await self._session.publicPostInfo(params={
            "type": "metaAndAssetCtxs",
        })
        meta = data[0]["universe"]
        asset_info = data[1]
        resp = {}
        for i, obj in enumerate(meta):
            name = obj["name"]  # e.g. "BTC"
            mark_price = float(asset_info[i]["markPx"])
            size_decimals = int(obj["szDecimals"])
            item = Market(
                internal_pair=name,
                base=name,
                quote="USD",
                price_precision=mark_price,
                contract_precision=1 / (10 ** size_decimals),
                min_contracts=1 / (10 ** size_decimals),
                coin_index=i,
                market_price=mark_price,
            )
            ext_pair = f"{name}/USDC:USDC"
            resp[ext_pair] = item
        self.market = resp

    async def close(self):
        await self._session.close()

    async def get_last_ohlcv(self, pair, timeframe, limit=1000) -> pd.DataFrame:
        # ... (implementation omitted)
        pass

    async def get_balance(self) -> UsdcBalance:
        data = await self._session.publicPostInfo(params={
            "type": "clearinghouseState",
            "user": self.public_address,
        })
        total = float(data["marginSummary"]["accountValue"])
        used = float(data["marginSummary"]["totalMarginUsed"])
        free = total - used
        return UsdcBalance(total=total, free=free, used=used)

    async def get_open_orders(self, symbols: List[str]) -> pd.DataFrame:
        orders_list = []

        async def fetch_orders(coin: str):
            symbol = self.pair_to_ext_pair(coin)
            try:
                # Include the wallet address as a user parameter in the API call.
                return await self._session.fetch_open_orders(symbol=symbol, params={"user": self.public_address})
            except Exception as e:
                print(f"Error fetching orders for {symbol}: {e}")
                return []

        orders_results = await asyncio.gather(*(fetch_orders(coin) for coin in symbols))

        for orders in orders_results:
            for order in orders:
                dt_str = order.get("datetime", None)
                if dt_str:
                    try:
                        dt_obj = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S.%fZ')
                    except ValueError:
                        dt_obj = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%SZ')
                    formatted_dt = dt_obj.strftime('%m/%d/%y - %H:%M:%S')
                else:
                    formatted_dt = "N/A"

                coin_name = order.get("info", {}).get("coin", "N/A")
                side = order.get("side", "").upper()
                order_type = order.get("type", "")
                price = float(order.get("price", 0))
                amount = float(order.get("amount", 0))
                value = price * amount

                orders_list.append({
                    "coin": coin_name,
                    "side": side,
                    "type": order_type,
                    "datetime": formatted_dt,
                    "price": price,
                    "amount": amount,
                    "value": value,
                })

        return pd.DataFrame(orders_list)

    def pair_to_ext_pair(self, coin: str) -> str:
        # Convert a coin symbol like 'BTC' to the Hyperliquid format 'BTC/USDC:USDC'
        return f"{coin}/USDC:USDC"

    async def get_open_positions(self, pairs: List[str]) -> pd.DataFrame:
        data = await self._session.publicPostInfo(params={
            "type": "clearinghouseState",
            "user": self.public_address,
        })
        if "assetPositions" not in data:
            return pd.DataFrame()

        pos_list = data["assetPositions"]
        rows = []
        for p in pos_list:
            position = p["position"]
            coin = position["coin"]
            ext_pair = self.pair_to_ext_pair(coin)
            if ext_pair not in pairs:
                continue

            szi = float(position["szi"])
            side = "long" if szi > 0 else "short"
            size_abs = abs(szi)
            usd_size = float(position["positionValue"])
            entry_price = float(position["entryPx"])
            value = entry_price * size_abs
            unrealizedPnl = float(position["unrealizedPnl"])

            open_timestamp = p.get("open_timestamp", 0)
            if open_timestamp:
                dt_obj = datetime.fromtimestamp(open_timestamp / 1000)
                dt_str = dt_obj.strftime('%m/%d/%y - %H:%M:%S')
            else:
                dt_str = "N/A"

            rows.append({
                "datetime": dt_str,
                "coin": coin,
                "side": side,
                "size": size_abs,
                "usd size": usd_size,
                "entry price": entry_price,
                "value": value,
                "unrealized p&l": unrealizedPnl,
            })

        return pd.DataFrame(rows)

    async def get_trade_history(self, coins: List[str], count: Union[int, str] = 10) -> pd.DataFrame:
        params = {
            "type": "userFills",
            "user": self.public_address,
            "aggregateByTime": True
        }
        data = await self._session.publicPostInfo(params=params)
        fills = data  # Assuming the API returns a list of fills.

        if coins:
            fills = [fill for fill in fills if fill.get("coin") in coins]

        fills = sorted(fills, key=lambda x: int(x.get("time", 0)), reverse=True)

        if isinstance(count, str) and count.lower() != "all":
            count = int(count)
        if isinstance(count, int):
            fills = fills[:count]

        trades_list = []
        for fill in fills:
            try:
                trade_id = fill.get("tid", "N/A")
                timestamp = int(fill.get("time", 0))
                dt_obj = datetime.fromtimestamp(timestamp / 1000)
                formatted_date = dt_obj.strftime('%m/%d/%y - %H:%M:%S')
                coin = fill.get("coin", "N/A")
                direction = fill.get("dir", "N/A")
                price = float(fill.get("px", 0))
                size = float(fill.get("sz", 0))
                notional = price * size
                fee = float(fill.get("fee", 0))
                closed_pnl = float(fill.get("closedPnl", 0))
                win_loss = "win" if closed_pnl > 0 else "loss" if closed_pnl < 0 else "neutral"
            except Exception as e:
                continue
            trades_list.append({
                "Trade ID": trade_id,
                "Date": formatted_date,
                "Coin": coin,
                "Direction": direction,
                "Price": price,
                "Size": size,
                "Notional": notional,
                "Fee": fee,
                "Closed PnL": closed_pnl,
                "Win/Loss": win_loss,
            })

        return pd.DataFrame(trades_list)
