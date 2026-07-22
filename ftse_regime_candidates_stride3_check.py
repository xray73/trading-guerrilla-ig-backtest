"""
ftse_regime_candidates_stride3_check.py — Controllo di robustezza sui 3
candidati regime FTSE100 (vedi ftse_regime_candidates_persistence.py).

MOTIVAZIONE: la finestra 21g campionata ogni settimana condivide ~14
dei 21 giorni tra una settimana e la successiva — per una correlazione
l'effetto è attutito, ma per ratio_dax_ftse_momentum (somma di
rendimenti su finestra) l'autocorrelazione tra settimane adiacenti è
quasi meccanica: se 14/21 giorni sono identici tra due finestre
consecutive, il segno della somma tende a restare lo stesso quasi per
costruzione, non necessariamente per un fenomeno di regime reale.

Questo script ripete ESATTAMENTE lo stesso protocollo (soglie fissate
sul train, applicate all'holdout senza ricalibrare) ma con stride=3
settimane: confronta lo stato alla settimana t con lo stato alla
settimana t+3 (~21 giorni di distanza reale, quasi zero sovrapposizione
residua nei dati sottostanti per la finestra 21g). Se la persistenza
crolla verso il base rate per un candidato, il risultato precedente era
in parte artefatto di sovrapposizione; se regge, è un fenomeno reale.

Nessuna modifica a market_regime_indicators o a research_*_candidates.
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
HOLDOUT_LABEL = "2024-2025"
STRIDE = 3  # settimane di distanza tra t e t+STRIDE (invece di 1)


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


def bucket_binary(values: pd.Series) -> pd.Series:
    return pd.Series(np.where(values >= 0, "su", "giu"), index=values.index)


def persistence_by_state_stride(states: pd.Series, stride: int) -> dict:
    """Come persistence_by_state ma confronta lo stato alla posizione i
    con quello alla posizione i+stride (invece di i+1) — riduce la
    sovrapposizione residua nei dati sottostanti a finestre lunghe."""
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


def run_candidate(name: str, magnitude_series: dict, is_directional: bool):
    print(f"\n{'=' * 70}\nCANDIDATO: {name} (stride={STRIDE} settimane)\n{'=' * 70}")

    for window_label, series in magnitude_series.items():
        weekly = weekly_nonoverlap(series)
        idx = weekly.index

        train_mask = in_periods(idx, [p for p in PERIODS if p != HOLDOUT_LABEL])
        holdout_mask = in_periods(idx, [HOLDOUT_LABEL])

        train_vals = weekly[train_mask]
        holdout_vals = weekly[holdout_mask]

        if len(train_vals) < 10 or len(holdout_vals) < 5:
            print(f"  [{window_label}] Dati insufficienti (train={len(train_vals)}, "
                  f"holdout={len(holdout_vals)}) — salto.")
            continue

        if is_directional:
            train_states = bucket_binary(train_vals)
            holdout_states = bucket_binary(holdout_vals)
            base_rate = 0.5
        else:
            q1, q2 = tercile_thresholds(train_vals)
            train_states = bucket_tercile(train_vals, q1, q2)
            holdout_states = bucket_tercile(holdout_vals, q1, q2)
            base_rate = 1 / 3

        train_persist = persistence_by_state_stride(train_states, STRIDE)
        holdout_persist = persistence_by_state_stride(holdout_states, STRIDE)

        print(f"\n  --- finestra {window_label} (base rate: {base_rate*100:.1f}%) ---")
        print(f"  train (n settimane={len(train_vals)}):")
        for state, (rate, n) in sorted(train_persist.items()):
            if np.isnan(rate):
                print(f"    {state:8s}: n=0")
            else:
                print(f"    {state:8s}: persistenza={rate*100:5.1f}% n={n} "
                      f"({(rate - base_rate) * 100:+.1f}pt vs base rate)")
        print(f"  holdout {HOLDOUT_LABEL} (n settimane={len(holdout_vals)}):")
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

    print("Scarico OHLC (DAX, FTSE100, GOLD, GBPUSD) via ohlc_data_source...")
    dax = get_ohlc("DAX", account_id, token).set_index("timestamp")["close"]
    ftse = get_ohlc("FTSE100", account_id, token).set_index("timestamp")["close"]
    gold = get_ohlc("GOLD", account_id, token).set_index("timestamp")["close"]
    gbpusd = get_ohlc("GBPUSD", account_id, token).set_index("timestamp")["close"]

    ret_ftse = ftse.pct_change()
    ret_gbpusd = gbpusd.pct_change()
    ret_gold = gold.pct_change()

    corr_ftse_gbpusd = {
        "7d": rolling_corr(ret_ftse, ret_gbpusd, "7D"),
        "21d": rolling_corr(ret_ftse, ret_gbpusd, "21D"),
    }
    run_candidate("corr_ftse_gbpusd (magnitude, terzili)", corr_ftse_gbpusd, is_directional=False)

    aligned = pd.concat([dax.rename("dax"), ftse.rename("ftse")], axis=1, sort=True).dropna()
    log_ratio = np.log(aligned["dax"] / aligned["ftse"])
    ratio_mom = {
        "7d": log_ratio.diff(1).rolling("7D").sum(),
        "21d": log_ratio.diff(1).rolling("21D").sum(),
    }
    run_candidate("ratio_dax_ftse_momentum (direzionale, binario)", ratio_mom, is_directional=True)

    corr_ftse_gold = {
        "7d": rolling_corr(ret_ftse, ret_gold, "7D"),
        "21d": rolling_corr(ret_ftse, ret_gold, "21D"),
    }
    run_candidate("corr_ftse_gold (magnitude, terzili)", corr_ftse_gold, is_directional=False)

    print("\n" + "=" * 70)
    print("Completato. Nessuna scrittura su D1 — solo stampa aggregata.")
    print("=" * 70)


if __name__ == "__main__":
    main()
