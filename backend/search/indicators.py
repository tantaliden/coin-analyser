"""Technische Indikatoren in pure Python. Keine Fallbacks — None wenn Baseline fehlt."""
import math


def sma(values, period):
    n = len(values)
    result = [None] * n
    for i in range(period - 1, n):
        result[i] = sum(values[i - period + 1:i + 1]) / period
    return result


def ema(values, period):
    n = len(values)
    result = [None] * n
    if n < period:
        return result
    k = 2 / (period + 1)
    seed = sum(values[:period]) / period
    result[period - 1] = seed
    for i in range(period, n):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


def rsi(closes, period=14):
    n = len(closes)
    result = [None] * n
    if n < period + 1:
        return result
    gains, losses = [0.0], [0.0]
    for i in range(1, n):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_gain = sum(gains[1:period + 1]) / period
    avg_loss = sum(losses[1:period + 1]) / period
    if avg_loss == 0:
        result[period] = 100.0 if avg_gain > 0 else 50.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    return result


def macd_hist(closes, fast=12, slow=26, signal=9):
    """MACD-Histogram = MACD-Line - Signal-Line."""
    n = len(closes)
    result = [None] * n
    ef = ema(closes, fast)
    es = ema(closes, slow)
    macd_line = [None] * n
    for i in range(n):
        if ef[i] is not None and es[i] is not None:
            macd_line[i] = ef[i] - es[i]
    valid = [(i, v) for i, v in enumerate(macd_line) if v is not None]
    if len(valid) < signal:
        return result
    vals = [v for _, v in valid]
    sig_vals = ema(vals, signal)
    for j, (i, _) in enumerate(valid):
        if sig_vals[j] is not None and macd_line[i] is not None:
            result[i] = macd_line[i] - sig_vals[j]
    return result


def bollinger_pos(closes, period=20, num_std=2):
    """Position im Band: -1 untere, 0 Mitte, +1 obere."""
    n = len(closes)
    result = [None] * n
    for i in range(period - 1, n):
        w = closes[i - period + 1:i + 1]
        mean = sum(w) / period
        var = sum((x - mean) ** 2 for x in w) / period
        std = math.sqrt(var)
        if std == 0:
            result[i] = 0.0
        else:
            result[i] = (closes[i] - mean) / (num_std * std)
    return result


def atr(candles, period=14):
    n = len(candles)
    result = [None] * n
    if n < period:
        return result
    trs = []
    for i, c in enumerate(candles):
        h, l = float(c['high']), float(c['low'])
        if i == 0:
            trs.append(h - l)
        else:
            pc = float(candles[i - 1]['close'])
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    result[period - 1] = sum(trs[:period]) / period
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + trs[i]) / period
    return result
