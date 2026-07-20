"""
analyze_adx_bucket_quality.py — Scompone i trade MR e V6 dei 5 periodi
ufficiali per fascia ADX, per rispondere a due domande poste in chat
il 19/07/2026:
  1) Ha senso una "banda morta" tra V6 (ADX>20) e MR (ADX<20)? Se i
     trade MR nella fascia 15-20 performano peggio di quelli sotto 15,
     sì — altrimenti la banda morta non aiuterebbe comunque.
  2) Come sono distribuiti i trade V6 per qualità/quantità man mano
     che l'ADX sale? Trend più forte = trade migliori, o è piatto?

METODO: riusa i motori standard INVARIATI (BacktestEngineFloatingKillSwitch
per V6, BacktestEngineMeanReversion per MR), nessuna modifica — pura
diagnostica sui trade già generati dai motori esistenti. Pooling di
tutti i trade DAX+FTSE100 sui 5 periodi ufficiali (necessario per
avere numeri abbastanza grandi da dire qualcosa, specialmente per MR).

Dati: legge da D1 via ohlc_data_source.py (aggiorna automaticamente le
barre mancanti da Dukascopy solo se necessario, come da convenzione
adottata oggi — nessun fetch pieno ripetuto).

Nessuna scrittura su D1 (oltre all'aggiornamento incrementale di
ohlc_prices via ohlc_data_source.py, se serve). Nessuna modifica a
engine.py, engine_mean_reversion.py o mean_reversion_signals.py.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_mean_reversion import BacktestEngineMeanReversion
from mean_reversion_signals import generate_mean_reversion_signals
from ohlc_data_source import get_ohlc

CAPITAL0 = 2000.0  # capitale pieno per entrambi i motori, per minimizzare distorsioni da size minima
SYMBOLS = ["DAX", "FTSE100"]

PERIODS = [
    ("2015-2016", "2015-01-05", "2016-12-29"),
    ("2020-covid", "2020-01-02", "2020-12-30"),
    ("2023", "2023-01-02", "2023-12-30"),
    ("2024-2025", "2024-01-03", "2025-12-31"),
    ("2026-ytd", "2026-01-05", "2026-07-10"),
]

MR_BUCKETS = [(0, 10), (10, 15), (15, 20)]
V6_BUCKETS = [(20, 25), (25, 30), (30, 35), (35, 40), (40, 200)]


def slice_period(df: pd.DataFrame, p_start: pd.Timestamp, p_end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["timestamp"] >= p_start) & (df["timestamp"] < p_end)].reset_index(drop=True)


def bucket_stats(trades_df: pd.DataFrame, buckets: list[tuple[float, float]], log) -> None:
    if trades_df.empty:
        log("  Nessun trade.")
        return
    for lo, hi in buckets:
        sub = trades_df[(trades_df["adx_at_entry"] >= lo) & (trades_df["adx_at_entry"] < hi)]
        if sub.empty:
            log(f"    ADX [{lo:.0f}-{hi:.0f}): n=0")
            continue
        wins = (sub["pnl"] > 0).sum()
        wr = 100 * wins / len(sub)
        pnl_sum = sub["pnl"].sum()
        expectancy = sub["pnl"].mean()
        log(f"    ADX [{lo:.0f}-{hi:.0f}): n={len(sub):4d}  WR={wr:5.1f}%  "
            f"PnL_totale={pnl_sum:+9.2f}  expectancy/trade={expectancy:+.2f}")


def main():
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=== Scomposizione trade V6 e MR per fascia ADX — 5 periodi ufficiali (pooled) ===\n")

    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        log("ERRORE: CLOUDFLARE_API_TOKEN o CLOUDFLARE_ACCOUNT_ID mancanti.")
        return

    log("Verifico/aggiorno OHLC (D1 + eventuali barre mancanti da Dukascopy)...")
    raw_full = {name: get_ohlc(name, account_id, token, log=log) for name in SYMBOLS}
    log("Fatto.\n")

    v6_signals_full = {name: eng.generate_signals(raw_full[name], eng.INSTRUMENTS[name]) for name in SYMBOLS}
    mr_signals_full = {name: generate_mean_reversion_signals(raw_full[name], eng.INSTRUMENTS[name], mode="rsi")
                        for name in SYMBOLS}

    all_v6_trades = []
    all_mr_trades = []

    for label, p_start_str, p_end_str in PERIODS:
        p_start = pd.Timestamp(p_start_str, tz="UTC")
        p_end = pd.Timestamp(p_end_str, tz="UTC") + pd.Timedelta(days=1)
        log(f"Periodo {label}...")

        v6_sig = {name: slice_period(v6_signals_full[name], p_start, p_end) for name in SYMBOLS}
        mr_sig = {name: slice_period(mr_signals_full[name], p_start, p_end) for name in SYMBOLS}

        eng_v6 = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
        trades_v6, _ = eng_v6.run(v6_sig)
        if not trades_v6.empty:
            all_v6_trades.append(trades_v6)

        eng_mr = BacktestEngineMeanReversion(capital0=CAPITAL0)
        trades_mr, _ = eng_mr.run(mr_sig)
        if not trades_mr.empty:
            all_mr_trades.append(trades_mr)

    v6_pooled = pd.concat(all_v6_trades, ignore_index=True) if all_v6_trades else pd.DataFrame()
    mr_pooled = pd.concat(all_mr_trades, ignore_index=True) if all_mr_trades else pd.DataFrame()

    log(f"\nTotale trade V6 pooled (5 periodi, DAX+FTSE100): {len(v6_pooled)}")
    log(f"Totale trade MR pooled (5 periodi, DAX+FTSE100): {len(mr_pooled)}\n")

    log("=" * 70)
    log("MEAN-REVERSION (ADX<20) — scomposizione per fascia, verifica banda morta")
    log("=" * 70)
    bucket_stats(mr_pooled, MR_BUCKETS, log)

    log("\n" + "=" * 70)
    log("V6 (ADX>20) — scomposizione per fascia, qualita'/quantita' vs forza del trend")
    log("=" * 70)
    bucket_stats(v6_pooled, V6_BUCKETS, log)

    # --- confronto diretto per la domanda "banda morta ha senso?" ---
    if not mr_pooled.empty:
        low_band = mr_pooled[mr_pooled["adx_at_entry"] < 15]
        high_band = mr_pooled[(mr_pooled["adx_at_entry"] >= 15) & (mr_pooled["adx_at_entry"] < 20)]
        log(f"\n{'='*70}\nCONFRONTO DIRETTO — MR sotto 15 vs MR 15-20\n{'='*70}")
        if not low_band.empty:
            log(f"ADX<15:   n={len(low_band):4d}  WR={100*(low_band['pnl']>0).sum()/len(low_band):.1f}%  "
                f"expectancy={low_band['pnl'].mean():+.2f}")
        else:
            log("ADX<15:   n=0 (campione insufficiente per qualunque conclusione)")
        if not high_band.empty:
            log(f"ADX 15-20: n={len(high_band):4d}  WR={100*(high_band['pnl']>0).sum()/len(high_band):.1f}%  "
                f"expectancy={high_band['pnl'].mean():+.2f}")
        else:
            log("ADX 15-20: n=0")

    os.makedirs("results", exist_ok=True)
    if not v6_pooled.empty:
        v6_pooled.to_csv("results/v6_trades_pooled.csv", index=False)
    if not mr_pooled.empty:
        mr_pooled.to_csv("results/mr_trades_pooled.csv", index=False)
    with open("results/analyze_adx_bucket_quality.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
