import numpy as np
from numba import njit
from strategy_helpers import *

def get_strategy() -> dict:
    return dict(
        name="ema_sma_adx_momentum_v2",
        variables=["ema_period", "sma_period", "adx_period", "atr_period", 
                   "atr_mult", "adx_threshold", "roc_period", "cmf_period",
                   "wait_buy", "wait_sell"],
        bounds=([20, 40, 10, 10, 0.5, 20, 5, 10, 10, 50],
                [60, 100, 30, 30, 2.5, 35, 20, 30, 150, 200]),
        simulate=simulate,
    )

def simulate(close, high, low, volume, x):
    ema_period = max(int(x[0]), 1)
    sma_period = max(int(x[1]), 1)
    adx_period = max(int(x[2]), 1)
    atr_period = max(int(x[3]), 1)
    atr_mult = float(x[4])
    adx_threshold = float(x[5])
    roc_period = max(int(x[6]), 1)
    cmf_period = max(int(x[7]), 1)
    wait_buy = int(x[8])
    wait_sell = int(x[9])
    
    ema = ema_np(close, ema_period)
    sma = sma_np(close, sma_period)
    adx_result = adx_np(high, low, close, adx_period)
    adx, pdi, mdi = adx_result[0], adx_result[1], adx_result[2]
    atr = atr_np(high, low, close, atr_period)
    roc = roc_np(close, roc_period)
    cmf = cmf_np(high, low, close, volume, cmf_period)
    
    return _execute(close, 1_000_000.0, ema, sma, adx, pdi, mdi, atr,
                    roc, cmf, atr_mult, adx_threshold,
                    wait_buy, wait_sell)

@njit(fastmath=True)
def _execute(close, start_cash, ema, sma, adx, pdi, mdi, atr,
             roc, cmf, atr_mult, adx_threshold, wait_buy, wait_sell):
    cash = start_cash
    num_coins = 0
    last_trade = 0
    num_trades = 0
    peak = 0.0
    
    for i in range(len(close)):
        # Skip NaN values from indicators
        if np.isnan(ema[i]) or np.isnan(sma[i]) or np.isnan(adx[i]) or np.isnan(roc[i]) or np.isnan(cmf[i]):
            continue
        
        price = close[i]
        
        # Track peak price for trailing stop
        if num_coins > 0 and price > peak:
            peak = price
        
        # TRAILING STOP: Exit if price drops from peak by ATR-based threshold
        # Only trigger if position has some profit buffer (price > peak * 0.98)
        if num_coins > 0 and peak > 0:
            profit_buffer = peak * 0.98
            if price >= profit_buffer:
                stop_price = peak * (1.0 - atr_mult * atr[i] / peak)
                if price <= stop_price:
                    cash, num_coins = sell_all(cash, num_coins, price)
                    last_trade = i
                    num_trades += 1
                    peak = 0.0
        
        # BUY signals: Multiple confirmations required
        # 1. EMA > SMA (trend up)
        # 2. ADX > threshold (strong trend)
        # 3. PDI > MDI (directional confirmation - bulls in control)
        # 4. ROC > 0 (momentum positive)
        # 5. CMF > 0 (volume supports price)
        # 6. Cooldown period since last trade
        if num_coins == 0:
            trend_up = ema[i] > sma[i]
            strong_trend = adx[i] > adx_threshold
            directional = pdi[i] > mdi[i]
            momentum = roc[i] > 0.0
            volume_support = cmf[i] > 0.0
            cooldown = i > last_trade + wait_buy
            
            if trend_up and strong_trend and directional and momentum and volume_support and cooldown:
                cash, num_coins = buy_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak = price
        
        # SELL signals: Trend reversal + cooldown
        # 1. EMA < SMA (trend down)
        # 2. Longer cooldown to let winners run
        elif num_coins > 0:
            trend_down = ema[i] < sma[i]
            cooldown_sell = i > last_trade + wait_sell
            
            if trend_down and cooldown_sell:
                cash, num_coins = sell_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak = 0.0
    
    # Force-sell at end to realize final P&L
    cash, num_coins = sell_all(cash, num_coins, close[-1])
    return cash / start_cash, num_trades