"""
engine_three_asset_gold_longonly_test.py — Sanity check + test causale del
filtro GOLD long-only (24/07/2026).

PASSO 1 (obbligatorio, non saltare): con gold_longonly_filter_active=False,
BacktestEngineV6GoldLongOnly deve produrre risultati IDENTICI a
BacktestEngineV6Gold su tutti e 5 i periodi ufficiali. Se anche un solo
trade differisce, il motore ha un bug e va corretto PRIMA di guardare il
passo 2 — stesso principio non-negoziabile di ogni sottoclasse nel progetto.

PASSO 2: confronto vero, gold_longonly_filter_active=True vs baseline
(BacktestEngineV6Gold, short incluso), sui 5 periodi ufficiali. Metriche:
PnL totale, R medio, numero trade, drawdown.

PASSO 3: bootstrap a blocchi di giornata (day-block, N=2000, stesso
protocollo di Regole_Backtest_MonteCarlo.md) sul delta PnL tra le due
varianti, per calibrare se la differenza osservata e' oltre il rumore.

Uso: python engine_three_asset_gold_longonly_test.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import engine as eng
from engine_three_asset_gold import BacktestEngineV6Gold, instruments_with_gold
from engine_three_asset_gold_longonly import BacktestEngineV6GoldLongOnly
from ohlc_data_source import get_ohlc
import os

CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")

PERIODS = {
    "2015-2016": ("2015-01-01", "2016-12-31"),
    "2020-covid": ("2020-01-01", "2020-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024-2025": ("2024-01-01", "2025-12-31"),
    "2026-ytd": ("2026-01-01", "2026-07-15"),
}

CAPITALE_INIZIALE = 2000.0
INSTRUMENTS = instruments_with_gold()


def load_data():
    print("Carico OHLC DAX/FTSE100/GOLD da D1 (cache incrementale)...")
    raw = {}
    for name in ["DAX", "FTSE100", "GOLD"]:
        raw[name] = get_ohlc(name, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, log=print)
    signals = {name: eng.generate_signals(raw[name], INSTRUMENTS[name]) for name in raw}
    return signals


def slice_period(signals: dict, start: str, end: str) -> dict:
    out = {}
    for name, df in signals.items():
        mask = (df["timestamp"] >= pd.Timestamp(start, tz="UTC")) & (df["timestamp"] <= pd.Timestamp(end, tz="UTC"))
        out[name] = df.loc[mask].reset_index(drop=True)
    return out


def run_sanity_check(signals: dict):
    print("\n" + "=" * 70)
    print("PASSO 1 — SANITY CHECK (parametro neutro deve riprodurre il baseline)")
    print("=" * 70)
    all_ok = True
    for label, (start, end) in PERIODS.items():
        data = slice_period(signals, start, end)

        eng_baseline = BacktestEngineV6Gold(capital0=CAPITALE_INIZIALE, instruments=INSTRUMENTS)
        trades_base, metrics_base = eng_baseline.run(data)

        eng_neutral = BacktestEngineV6GoldLongOnly(
            capital0=CAPITALE_INIZIALE, instruments=INSTRUMENTS,
            gold_longonly_filter_active=False,
        )
        trades_neutral, metrics_neutral = eng_neutral.run(data)

        n_base, n_neutral = len(trades_base), len(trades_neutral)
        pnl_base = float(metrics_base["pnl_totale"].iloc[0]) if "pnl_totale" in metrics_base else eng_baseline.capital - CAPITALE_INIZIALE
        pnl_neutral = float(metrics_neutral["pnl_totale"].iloc[0]) if "pnl_totale" in metrics_neutral else eng_neutral.capital - CAPITALE_INIZIALE

        identical = (n_base == n_neutral) and abs(pnl_base - pnl_neutral) < 0.01
        status = "OK" if identical else "FALLITO — DIFFERENZA RILEVATA, NON PROCEDERE"
        print(f"  {label}: baseline n_trade={n_base} pnl={pnl_base:.2f}  |  "
              f"neutro n_trade={n_neutral} pnl={pnl_neutral:.2f}  ->  {status}")
        if not identical:
            all_ok = False

    if not all_ok:
        raise SystemExit("\nSANITY CHECK FALLITO — correggere il motore prima di qualunque test causale.")
    print("\nSanity check PASS su tutti e 5 i periodi. Procedo al test causale.\n")


def run_causal_test(signals: dict):
    print("=" * 70)
    print("PASSO 2 — CONFRONTO: baseline (GOLD long+short) vs GOLD long-only")
    print("=" * 70)

    results = []
    for label, (start, end) in PERIODS.items():
        data = slice_period(signals, start, end)

        eng_baseline = BacktestEngineV6Gold(capital0=CAPITALE_INIZIALE, instruments=INSTRUMENTS)
        trades_base, _ = eng_baseline.run(data)
        pnl_base = eng_baseline.capital - CAPITALE_INIZIALE

        eng_filtered = BacktestEngineV6GoldLongOnly(
            capital0=CAPITALE_INIZIALE, instruments=INSTRUMENTS,
            gold_longonly_filter_active=True,
        )
        trades_filt, _ = eng_filtered.run(data)
        pnl_filt = eng_filtered.capital - CAPITALE_INIZIALE

        delta = pnl_filt - pnl_base
        print(f"\n--- {label} ---")
        print(f"  Baseline (GOLD long+short): {len(trades_base)} trade, PnL {pnl_base:+.2f}EUR")
        print(f"  Filtrato (GOLD long-only):  {len(trades_filt)} trade, PnL {pnl_filt:+.2f}EUR")
        print(f"  Delta: {delta:+.2f}EUR")

        results.append({
            "periodo": label, "n_trade_base": len(trades_base), "pnl_base": pnl_base,
            "n_trade_filt": len(trades_filt), "pnl_filt": pnl_filt, "delta": delta,
            "trades_base": trades_base, "trades_filt": trades_filt,
        })

    return results


def bootstrap_delta(results: list, n_iter: int = 2000, seed: int = 42):
    """Bootstrap a blocchi di giornata sul delta di PnL, aggregato sui 5
    periodi. Stesso protocollo standard del progetto (Regole_Backtest_
    MonteCarlo.md): resampling per giorno, non per singolo trade, per non
    rompere la dipendenza tra trade dello stesso giorno."""
    print("\n" + "=" * 70)
    print("PASSO 3 — BOOTSTRAP a blocchi di giornata sul delta (N=2000)")
    print("=" * 70)

    rng = np.random.default_rng(seed)

    all_days_base = {}
    all_days_filt = {}
    for r in results:
        tb = r["trades_base"]
        tf = r["trades_filt"]
        if len(tb):
            tb = tb.copy()
            tb["day"] = pd.to_datetime(tb["entry_time"]).dt.date.astype(str)
        if len(tf):
            tf = tf.copy()
            tf["day"] = pd.to_datetime(tf["entry_time"]).dt.date.astype(str)
        for day, grp in (tb.groupby("day") if len(tb) else []):
            all_days_base.setdefault(day, []).extend(grp["pnl"].tolist())
        for day, grp in (tf.groupby("day") if len(tf) else []):
            all_days_filt.setdefault(day, []).extend(grp["pnl"].tolist())

    days = sorted(set(all_days_base) | set(all_days_filt))
    observed_delta = sum(sum(v) for v in all_days_filt.values()) - sum(sum(v) for v in all_days_base.values())

    null_deltas = []
    for _ in range(n_iter):
        sampled_days = rng.choice(days, size=len(days), replace=True)
        d_base = sum(sum(all_days_base.get(d, [])) for d in sampled_days)
        d_filt = sum(sum(all_days_filt.get(d, [])) for d in sampled_days)
        null_deltas.append(d_filt - d_base)

    null_deltas = np.array(null_deltas)
    z = (observed_delta - null_deltas.mean()) / null_deltas.std()
    ci_low, ci_high = np.percentile(null_deltas, [2.5, 97.5])
    pct_le_zero = (null_deltas <= 0).mean() if observed_delta > 0 else (null_deltas >= 0).mean()

    print(f"\nDelta osservato (filtrato - baseline), aggregato 5 periodi: {observed_delta:+.2f} EUR")
    print(f"IC 95% bootstrap: [{ci_low:+.2f}, {ci_high:+.2f}] EUR")
    print(f"Z-score: {z:.3f}")
    print(f"Frazione iterazioni bootstrap con segno opposto/nullo: {pct_le_zero*100:.1f}%")
    print(f"\nVerdetto: {'PROMOSSO (z>=2, IC esclude lo zero)' if abs(z) >= 2 and (ci_low>0 or ci_high<0) else 'NON PROMOSSO / AMBIGUO'}")


def main():
    if not CLOUDFLARE_ACCOUNT_ID or not CLOUDFLARE_API_TOKEN:
        print("ERRORE: CLOUDFLARE_ACCOUNT_ID o CLOUDFLARE_API_TOKEN mancanti.")
        return
    signals = load_data()
    run_sanity_check(signals)
    results = run_causal_test(signals)
    bootstrap_delta(results)


if __name__ == "__main__":
    main()
