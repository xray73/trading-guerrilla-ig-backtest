"""
engine_atr_regime.py — Modulazione del rischio sui 3 slot BASE
(non solo extra) in funzione del regime di volatilità corrente,
misurato come percentile causale dell'ATR rispetto a una finestra
rolling di N giorni passati (mai barre future — stesso principio
già applicato ovunque nel progetto, es. EMA/ADX/ATR stessi).

Costruita sopra BacktestEngineFloatingKillSwitch (motore standard
adottato il 15/07/2026), non sopra BacktestEngineExtendedOrders —
test isolato su una sola variabile alla volta (i 3 slot base), la
combinazione con gli slot extra è un test separato successivo se
questo dovesse promuovere.

Le fasce (tercili: basso/medio/alto percentile) sono strutturalmente
fisse per costruzione (0-33°, 33-66°, 66-100°) — quello che va
calibrato via grid search è il MOLTIPLICATORE di rischio per ciascuna
fascia, non i confini delle fasce stesse.

Sanity check: con tutti e 3 i moltiplicatori a 1.0 (nessuna
modulazione), deve essere identica a BacktestEngineFloatingKillSwitch.
"""

from __future__ import annotations

import dataclasses
import pandas as pd
import numpy as np

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch


def compute_atr_regime(df: pd.DataFrame, window_days: int, bars_per_day_estimate: int = 45) -> pd.DataFrame:
    """Aggiunge 'atr_pctile' (percentile causale, 0-1) e 'atr_tier'
    ('low'/'medium'/'high') al DataFrame. Causale: ogni barra usa SOLO
    ATR di barre precedenti (mai la propria barra futura inclusa oltre
    se stessa, finestra chiusa a sinistra fino alla barra corrente)."""
    out = df.copy().reset_index(drop=True)
    window_bars = max(window_days * bars_per_day_estimate, 50)

    def rolling_percentile(window: np.ndarray) -> float:
        if len(window) < 10 or np.isnan(window[-1]):
            return np.nan
        current = window[-1]
        valid = window[~np.isnan(window)]
        if len(valid) < 10:
            return np.nan
        return float((valid < current).sum()) / len(valid)

    out["atr_pctile"] = out["atr"].rolling(window=window_bars, min_periods=50).apply(
        rolling_percentile, raw=True)

    def tier_from_pctile(p):
        if pd.isna(p):
            return "medium"  # default prudente finché la finestra non si riempie
        if p < 0.3333:
            return "low"
        if p < 0.6667:
            return "medium"
        return "high"

    out["atr_tier"] = out["atr_pctile"].apply(tier_from_pctile)
    return out


class BacktestEngineATRRegime(BacktestEngineFloatingKillSwitch):

    def __init__(self, *args, tier_multipliers: dict[str, float] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tier_multipliers = tier_multipliers or {"low": 1.0, "medium": 1.0, "high": 1.0}
        self.tier_open_counts = {"low": 0, "medium": 0, "high": 0}

    def _open_position(self, instrument: str, direction: str, bar: pd.Series,
                        atr_at_entry: float, adx_at_entry: float):
        original_inst = self.instruments[instrument]
        tier = bar["atr_tier"] if "atr_tier" in bar.index else "medium"
        multiplier = self.tier_multipliers.get(tier, 1.0)

        modulated_inst = dataclasses.replace(original_inst, risk_pct=original_inst.risk_pct * multiplier)
        self.instruments[instrument] = modulated_inst
        n_before = len(self.open_positions)
        try:
            super()._open_position(instrument, direction, bar, atr_at_entry, adx_at_entry)
        finally:
            self.instruments[instrument] = original_inst

        if len(self.open_positions) > n_before:
            self.tier_open_counts[tier] = self.tier_open_counts.get(tier, 0) + 1
