"""Konfigurierbare Werte fuer den Predictor/Counter-Search. Keine Hardcodes im Scanner."""

SCAN_SETTINGS = {
    "max_period_days": 365,
    "default_period_days": 90,
    "max_results": 500,
    "per_criterion_min_score": 0.5,
    "batch_size_symbols": 50,
}

INITIAL_POINT_SETTINGS = {
    "window_minutes": 30,
    "max_points": 10,
    "default_match_mode": "all",
}

PATTERN_THRESHOLDS = {
    "doji_body_ratio": 0.05,
    "hammer_wick_ratio": 2.0,
    "engulfing_min_body": 0.01,
    "marubozu_wick_ratio": 0.02,
    "tweezer_tolerance": 0.001,
    "spinning_top_body_ratio": 0.3,
    "piercing_min_penetration": 0.5,
}

FUZZY_DEFAULTS = {
    "value_tolerance_pct": 10.0,
    "time_tolerance_min": 10.0,
    "slope_tolerance_pct": 15.0,
    "ratio_tolerance_pct": 10.0,
}

AGG_TABLE_MAP = {
    1: "klines_1m", 5: "klines_5m", 15: "klines_15m", 30: "klines_30m",
    60: "klines_1h", 240: "klines_4h", 1440: "klines_1d",
}

# Defaults fuer Indicator-Set Erstellung (kein Hardcode im Endpoint-Code)
SET_DEFAULTS = {
    "duration_minutes": 120,
    "direction": "up",
    "target_percent": 5.0,
    "prehistory_minutes": 720,
    "candle_timeframe": 1,
    "min_accuracy": 70.0,
    "take_profit_pct": 5.0,
    "stop_loss_pct": 3.0,
}

# Aggregator-Strings fuer DB-Spalten (aggregator-Spalte in indicator_items)
AGGREGATOR_STRINGS = {
    1: "1m", 2: "2m", 5: "5m", 10: "10m", 15: "15m", 30: "30m",
    60: "1h", 120: "2h", 240: "4h",
}

# Auto-Anomaly-Detection (Rolling-Window z-score)
ANOMALY_SETTINGS = {
    "rolling_window_candles": 60,      # Vergangene Candles als Baseline
    "min_z_score": 2.0,                 # Ab hier 'anomal'
    "strong_z_score": 3.0,              # Ab hier 'stark anomal'
    "max_suggestions": 50,              # Max Vorschlaege pro Chart
    "metrics": [                        # Alle Metriken die gescannt werden
        "volume", "trades", "body_pct", "range_pct",
        "upper_wick_pct", "lower_wick_pct", "close_delta_pct",
        "rsi_14", "macd_hist", "bollinger_pos", "atr_14",
    ],
    "metric_weights": {                 # Gewichtung im Gesamt-Score
        "volume": 1.2, "trades": 1.0, "body_pct": 0.9, "range_pct": 0.8,
        "upper_wick_pct": 0.7, "lower_wick_pct": 0.7, "close_delta_pct": 1.0,
        "rsi_14": 1.0, "macd_hist": 1.0, "bollinger_pos": 0.9, "atr_14": 0.8,
    },
}

# Window-basierte Metriken: Summen, %-Spruenge, Slopes ueber N Candles
WINDOW_METRIC_SETTINGS = {
    "window_sizes": [5, 10, 15, 30],
    "families": {
        "volume_sum":  {"source": "volume",    "op": "sum",   "weight": 1.3},
        "trades_sum":  {"source": "trades",    "op": "sum",   "weight": 1.1},
        "close_pct":   {"source": "close",     "op": "pct",   "weight": 1.2},
        "rsi_slope":   {"source": "rsi_14",    "op": "slope", "weight": 1.0},
        "macd_slope":  {"source": "macd_hist", "op": "slope", "weight": 1.0},
    },
}

# ANOMALY_SETTINGS.metrics und metric_weights automatisch um Window-Familien erweitern
for _fam_name, _fam_cfg in WINDOW_METRIC_SETTINGS["families"].items():
    for _w in WINDOW_METRIC_SETTINGS["window_sizes"]:
        _key = f"{_fam_name}_{_w}"
        ANOMALY_SETTINGS["metrics"].append(_key)
        ANOMALY_SETTINGS["metric_weights"][_key] = _fam_cfg["weight"]

# Alle Candle-Muster die im Batch-Scan erkannt werden
ALL_PATTERN_IDS = [
    "doji", "hammer", "inverted_hammer", "spinning_top",
    "marubozu_bull", "marubozu_bear",
    "engulfing_bull", "engulfing_bear",
    "harami_bull", "harami_bear",
    "piercing", "dark_cloud",
    "tweezer_top", "tweezer_bottom",
    "three_white", "three_black",
    "morning_star", "evening_star",
]

# Zeit-Fenster-Scoring (Slope mit Dauer-Unschaerfe)
WINDOW_SCAN_SETTINGS = {
    "duration_step_candles": 1,         # Schrittweite bei Dauer-Variation
    "position_step_candles": 1,         # Schrittweite bei Positions-Variation
}

# Batch-Anomalie-Aggregation ueber alle Suchergebnisse
BATCH_ANOMALY_SETTINGS = {
    "bucket_minutes": 5,                # Offset-Bucket-Breite (Minuten vor Event)
}

# Frequent-Itemset-Mining fuer Anomalie-Kombinationen
ITEMSET_SETTINGS = {
    "default_bucket_minutes": 5,
    "default_min_support_pct": 50.0,
    "default_min_set_size": 2,
    "default_max_set_size": 5,
    "hard_max_set_size": 10,              # absolute Obergrenze (Sicherheit)
    "max_candidates_per_level": 200000,   # Abbruch wenn Kombinationsraum explodiert
}
