"""
ftse_gold_crossval_covid_holdout.py — Doppia validazione incrociata per
corr_ftse_gold (unico candidato sopravvissuto al controllo stride3, vedi
ftse_regime_candidates_stride3_check.py). Stesso principio già usato per
il regime composito ATR%+correlazione DAX-FTSE100 il 22/07 (train-
esclude-2020 vs holdout-solo-2020): se un pattern regge sia con holdout
2024-2025 SIA con holdout 2020-covid (periodi indipendenti, calibrati
separatamente), è più difficile che sia rumore centrato su un solo
periodo.

Qui: train = tutti i periodi TRANNE 2020-covid (incluso 2024-2025),
holdout = SOLO 2020-covid. Soglie terzili fissate solo sul train.
Protocollo stride3 (quello sopravvissuto al controllo di robustezza)
come metodo primario; stride1 riportato per confronto diretto con i
risultati già ottenuti.

Nessuna scrittura su D1 — solo stampa aggregata.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ohlc_data_source import get_ohlc

PERIODS = {
    "2015-2016": ("2015-01-01", "2017-01-01"),
    "2020-covid": ("2020-01-01", "2021-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024-2025": ("2024-01-01", "2026-01-01"),
    "2026-ytd": ("2026-01-01", "2026-07-14"),
}
HOLDOUT_LABEL = "2020-covid"  # <-- differenza chiave rispetto allo script precedente


def in_periods(ts_index: pd.DatetimeIndex, labels: list[str]) -> pd.Series:
    mask = pd.Series(False, index=ts_index)
    for lbl in labels:
        start, end = PERIODS[lbl]
        mask |= (ts_index >= pd.Timestamp(start, tz="UTC")) & (ts_index < pd.Timestamp(end, tz="UTC"))
    return mask


def rolling_corr(a: pd.Series, b: pd.Series, window: str) -> pd.Series:
    aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1, sort=True).dropna()
    return aligned["a"].rolling(window).corr(aligned["b"]).reindex(a.index)


def weekly_nonoverlap(series: pd.Series) -> pd.Series:
    return series.resample("W").last().dropna()


def tercile_thresholds(train_values: pd.Series) -> tuple[float, float]:
    q1, q2 = train_values.quantile([1 / 3, 2 / 3])
    return q1, q2


def bucket_tercile(values: pd.Series, q1: float, q2: float) -> pd.Series:
    return pd.cut(values, bins=[-np.inf, q1, q2, np.inf], labels=["bassa", "media", "alta"])


def persistence_by_state_stride(states: pd.Series, stride: int) -> dict:
    s = states.dropna()
    if len(s) < stride + 2:
        return {}
    cur = s.iloc[:-stride].values
    nxt = s.iloc[stride:].values
    out = {}
    categories = s.cat.categories if hasattr(s, "cat") else sorted(set(s))
    for state in categories:
        mask = cur == state
        n = mask.sum()
        if n == 0:
            out[state] = (np.nan, 0)
            continue
        persist_rate = (nxt[mask] == state).mean()
        out[state] = (persist_rate, int(n))
    return out


def run_window(window_label: str, series: pd.Series, stride: int):
    weekly = weekly_nonoverlap(series)
    idx = weekly.index

    train_mask = in_periods(idx, [p for p in PERIODS if p != HOLDOUT_LABEL])
    holdout_mask = in_periods(idx, [HOLDOUT_LABEL])

    train_vals = weekly[train_mask]
    holdout_vals = weekly[holdout_mask]

    if len(train_vals) < 10 or len(holdout_vals) < 5:
        print(f"  [{window_label}, stride={stride}] Dati insufficienti "
              f"(train={len(train_vals)}, holdout={len(holdout_vals)}) — salto.")
        return

    q1, q2 = tercile_thresholds(train_vals)
    train_states = bucket_tercile(train_vals, q1, q2)
    holdout_states = bucket_tercile(holdout_vals, q1, q2)
    base_rate = 1 / 3

    train_persist = persistence_by_state_stride(train_states, stride)
    holdout_persist = persistence_by_state_stride(holdout_states, stride)

    print(f"\n  --- finestra {window_label}, stride={stride} settimane "
          f"(base rate: {base_rate*100:.1f}%) ---")
    print(f"  train esclude 2020-covid (n settimane={len(train_vals)}):")
    for state, (rate, n) in sorted(train_persist.items()):
        if np.isnan(rate):
            print(f"    {state:8s}: n=0")
        else:
            print(f"    {state:8s}: persistenza={rate*100:5.1f}% n={n} "
                  f"({(rate - base_rate) * 100:+.1f}pt vs base rate)")
    print(f"  holdout SOLO 2020-covid (n settimane={len(holdout_vals)}):")
    for state, (rate, n) in sorted(holdout_persist.items()):
        if np.isnan(rate):
            print(f"    {state:8s}: n=0")
        else:
            print(f"    {state:8s}: persistenza={rate*100:5.1f}% n={n} "
                  f"({(rate - base_rate) * 100:+.1f}pt vs base rate)")


def main():
    import os
    account_id = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    token = os.environ["CLOUDFLARE_API_TOKEN"]

    print("Scarico OHLC (FTSE100, GOLD) via ohlc_data_source...")
    ftse = get_ohlc("FTSE100", account_id, token).set_index("timestamp")["close"]
    gold = get_ohlc("GOLD", account_id, token).set_index("timestamp")["close"]

    ret_ftse = ftse.pct_change()
    ret_gold = gold.pct_change()

    corr_7d = rolling_corr(ret_ftse, ret_gold, "7D")
    corr_21d = rolling_corr(ret_ftse, ret_gold, "21D")

    print(f"\n{'=' * 70}\nCANDIDATO: corr_ftse_gold — holdout alternativo 2020-covid\n{'=' * 70}")
    for window_label, series in (("7d", corr_7d), ("21d", corr_21d)):
        run_window(window_label, series, stride=1)
        run_window(window_label, series, stride=3)

    print("\n" + "=" * 70)
    print("Completato. Nessuna scrittura su D1 — solo stampa aggregata.")
    print("=" * 70)


if __name__ == "__main__":
    main()
