"""
compare_persistence_variants.py — Confronta il motore V6 standard
(persistenza minima=1, comportamento attuale) contro varianti che
richiedono che il segnale sia rimasto attivo per almeno 2/3/4 barre
prima di aprire una posizione, sui 5 periodi ufficiali.

Origine: osservazione sul dataset di ricerca — i pochi trade (9% del
totale) entrati per caso con persistenza>=2 avevano R medio ~3x
superiore. Questo test verifica se aspettare DELIBERATAMENTE produce
davvero un vantaggio, o se l'osservazione originale era un artefatto
di selezione (i trade a persistenza alta nel dataset originale erano
quelli "sopravvissuti" ad altre condizioni, non un campione neutro).

Motore: engine_persistence_variant.py (sottoclasse isolata di
BacktestEngineFloatingKillSwitch, sanity check gia' passato in locale:
persistenza=1 produce risultati IDENTICI al motore standard).

Nessuna modifica a engine.py o alle sottoclassi esistenti. Nessuna
scrittura su trades/backtest_runs/live_*/research_v6_*. Output SOLO
aggregato per periodo + totale (num_trades, win_rate, profit_factor,
expectancy, max_drawdown, pnl_totale) — nessun dump di trade individuali.
"""

from __future__ import annotations

import os
import pandas as pd

import engine as eng
from engine_persistence_variant import BacktestEnginePersistenceVariant

CAPITAL0 = 2000.0
SYMBOLS = ["DAX", "FTSE100"]
MIN_PERSISTENCE_VALUES = [1, 2, 3, 4]

PERIODS = [
    ("2015-2016", "2015-01-05", "2016-12-29"),
    ("2020-covid", "2020-01-02", "2020-12-30"),
    ("2023", "2023-01-02", "2023-12-30"),
    ("2024-2025", "2024-01-03", "2025-12-31"),
    ("2026-ytd", "2026-01-05", None),
]


def slice_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)


def main():
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id or not token:
        print("ERRORE: CLOUDFLARE_ACCOUNT_ID o CLOUDFLARE_API_TOKEN mancanti.")
        return

    from ohlc_data_source import get_ohlc

    print("Scarico/aggiorno storico DAX/FTSE100...")
    raw = {name: get_ohlc(name, account_id, token, log=print) for name in SYMBOLS}
    signals_full = {name: eng.generate_signals(raw[name], eng.INSTRUMENTS[name]) for name in SYMBOLS}
    print("Fatto.\n")

    rows = []
    for label, start_str, end_str in PERIODS:
        start = pd.Timestamp(start_str, tz="UTC")
        end = pd.Timestamp(end_str, tz="UTC") + pd.Timedelta(days=1) if end_str else pd.Timestamp.now(tz="UTC")
        print(f"--- Periodo {label} ---")

        sig_period = {name: slice_period(signals_full[name], start, end) for name in SYMBOLS}

        for min_p in MIN_PERSISTENCE_VALUES:
            engine_run = BacktestEnginePersistenceVariant(
                capital0=CAPITAL0, instruments=eng.INSTRUMENTS, min_persistence=min_p)
            trades_df, metrics_df = engine_run.run(sig_period)
            m = metrics_df.iloc[0]
            print(f"  persistenza>={min_p}: {m['num_trades']} trade, "
                  f"win_rate={m['win_rate']*100:.1f}%, PF={m['profit_factor']:.2f}, "
                  f"pnl={m['pnl_total']:+.2f}")
            rows.append({
                "periodo": label, "min_persistence": min_p,
                "num_trades": int(m["num_trades"]), "win_rate": m["win_rate"],
                "profit_factor": m["profit_factor"] if m["profit_factor"] != float("inf") else None,
                "expectancy": m["expectancy"], "max_drawdown_pct": m["max_drawdown_pct"],
                "pnl_total": m["pnl_total"],
            })

    df = pd.DataFrame(rows)
    print(f"\n{'='*70}\nAGGREGATO SUI 5 PERIODI, per soglia di persistenza\n{'='*70}")
    for min_p in MIN_PERSISTENCE_VALUES:
        subset = df[df["min_persistence"] == min_p]
        total_trades = subset["num_trades"].sum()
        total_pnl = subset["pnl_total"].sum()
        avg_win_rate = (subset["win_rate"] * subset["num_trades"]).sum() / total_trades if total_trades else float("nan")
        print(f"persistenza>={min_p}: {total_trades} trade totali, "
              f"pnl totale={total_pnl:+.2f} EUR, win_rate medio pesato={avg_win_rate*100:.1f}%")

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/compare_persistence_variants.csv", index=False)
    print("\nDettaglio per periodo salvato in results/compare_persistence_variants.csv")


if __name__ == "__main__":
    main()
