"""
calibrate_mr_adx_band.py — Verifica se restringere il mean-reversion
alla banda ADX [15-20) (escludendo <15) migliora il risultato rispetto
al comportamento attuale (attivo su tutto ADX<20).

NOTA METODOLOGICA IMPORTANTE: la soglia 15 è stata scoperta guardando
il pooling di TUTTI i 5 periodi ufficiali insieme (analyze_adx_bucket_quality.py,
19/07/2026) — quindi non esiste un vero periodo "mai visto" su cui fare
un train/test classico (sarebbe circolare, la soglia è già "vista" su
tutti loro). L'approccio qui è diverso e più onesto in questo caso
specifico: verifica PERIODO PER PERIODO (non solo in aggregato) se il
vantaggio è diffuso o dipende da 1-2 periodi — coerente col principio 3
del Protocollo Anti-Rumore (robustezza), non un vero test fuori
campione.

METODO: stesso motore BacktestEngineMeanReversion INVARIATO — la banda
è implementata a livello di segnale (pre-filtro: dove adx<15, il
segnale viene azzerato PRIMA di passare i dati al motore), non una
modifica al motore stesso.

Dati letti da D1 via ohlc_data_source.py (aggiorna solo le barre
mancanti). Nessuna scrittura su D1 oltre l'aggiornamento incrementale.
Nessuna modifica a engine.py, engine_mean_reversion.py o
mean_reversion_signals.py.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import engine as eng
from engine_mean_reversion import BacktestEngineMeanReversion
from mean_reversion_signals import generate_mean_reversion_signals
from ohlc_data_source import get_ohlc

CAPITAL0 = 2000.0
SYMBOLS = ["DAX", "FTSE100"]
ADX_BAND_MIN = 15.0  # soglia scoperta il 19/07/2026, verificata qui period per periodo

PERIODS = [
    ("2015-2016", "2015-01-05", "2016-12-29"),
    ("2020-covid", "2020-01-02", "2020-12-30"),
    ("2023", "2023-01-02", "2023-12-30"),
    ("2024-2025", "2024-01-03", "2025-12-31"),
    ("2026-ytd", "2026-01-05", "2026-07-10"),
]


def slice_period(df: pd.DataFrame, p_start: pd.Timestamp, p_end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["timestamp"] >= p_start) & (df["timestamp"] < p_end)].reset_index(drop=True)


def apply_adx_band(signals: pd.DataFrame, adx_min: float) -> pd.DataFrame:
    """Azzera il segnale dove adx < adx_min — il motore non modifica,
    e' il segnale in ingresso che viene ristretto."""
    out = signals.copy()
    out.loc[out["adx"] < adx_min, "signal"] = None
    return out


def metrics_summary(trades_df: pd.DataFrame, capital0: float) -> dict:
    if trades_df.empty:
        return {"n_trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan,
                "pnl_total": 0.0, "max_drawdown_pct": 0.0, "expectancy": np.nan}
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    sum_wins, sum_losses = wins["pnl"].sum(), losses["pnl"].sum()
    pf = sum_wins / abs(sum_losses) if sum_losses != 0 else np.inf
    equity = capital0 + trades_df["pnl"].cumsum()
    running_max = equity.cummax()
    dd = ((equity - running_max) / running_max).min() * 100
    return {
        "n_trades": len(trades_df), "win_rate_pct": 100 * len(wins) / len(trades_df),
        "profit_factor": pf, "pnl_total": trades_df["pnl"].sum(),
        "max_drawdown_pct": dd, "expectancy": trades_df["pnl"].mean(),
    }


def main():
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=== Verifica banda ADX [15-20) per mean-reversion — period per periodo ===\n")

    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        log("ERRORE: CLOUDFLARE_API_TOKEN o CLOUDFLARE_ACCOUNT_ID mancanti.")
        return

    log("Verifico/aggiorno OHLC (D1 + eventuali barre mancanti da Dukascopy)...")
    raw_full = {name: get_ohlc(name, account_id, token, log=log) for name in SYMBOLS}
    log("Fatto.\n")

    mr_signals_full = {name: generate_mean_reversion_signals(raw_full[name], eng.INSTRUMENTS[name], mode="rsi")
                        for name in SYMBOLS}
    mr_signals_banded = {name: apply_adx_band(mr_signals_full[name], ADX_BAND_MIN) for name in SYMBOLS}

    rows = []
    for label, p_start_str, p_end_str in PERIODS:
        p_start = pd.Timestamp(p_start_str, tz="UTC")
        p_end = pd.Timestamp(p_end_str, tz="UTC") + pd.Timedelta(days=1)
        log(f"Periodo {label}")

        sig_base = {name: slice_period(mr_signals_full[name], p_start, p_end) for name in SYMBOLS}
        sig_band = {name: slice_period(mr_signals_banded[name], p_start, p_end) for name in SYMBOLS}

        eng_base = BacktestEngineMeanReversion(capital0=CAPITAL0)
        trades_base, _ = eng_base.run(sig_base)
        m_base = metrics_summary(trades_base, CAPITAL0)

        eng_band = BacktestEngineMeanReversion(capital0=CAPITAL0)
        trades_band, _ = eng_band.run(sig_band)
        m_band = metrics_summary(trades_band, CAPITAL0)

        log(f"  Attuale (ADX<20 tutto):  n={m_base['n_trades']:3d}  WR={m_base['win_rate_pct']:.1f}%  "
            f"PF={m_base['profit_factor']:.2f}  PnL={m_base['pnl_total']:+.2f}  maxDD={m_base['max_drawdown_pct']:.1f}%")
        log(f"  Banda [15-20) soltanto:  n={m_band['n_trades']:3d}  WR={m_band['win_rate_pct']:.1f}%  "
            f"PF={m_band['profit_factor']:.2f}  PnL={m_band['pnl_total']:+.2f}  maxDD={m_band['max_drawdown_pct']:.1f}%")
        log(f"  Delta PnL: {m_band['pnl_total'] - m_base['pnl_total']:+.2f}\n")

        rows.append({
            "periodo": label,
            "base_n": m_base["n_trades"], "base_pnl": m_base["pnl_total"], "base_dd": m_base["max_drawdown_pct"],
            "band_n": m_band["n_trades"], "band_pnl": m_band["pnl_total"], "band_dd": m_band["max_drawdown_pct"],
            "delta": m_band["pnl_total"] - m_base["pnl_total"],
        })

    summary_df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv("results/calibrate_mr_adx_band.csv", index=False)

    log(f"{'='*70}\nRIEPILOGO\n{'='*70}")
    log(f"PnL totale — attuale: {summary_df['base_pnl'].sum():+.2f}  "
        f"banda [15-20): {summary_df['band_pnl'].sum():+.2f}  "
        f"delta: {summary_df['band_pnl'].sum()-summary_df['base_pnl'].sum():+.2f}")
    log(f"Periodi in cui la banda migliora: {(summary_df['delta'] > 0).sum()}/5")
    log(f"Periodi in cui la banda peggiora: {(summary_df['delta'] < 0).sum()}/5")
    log(f"Drawdown medio — attuale: {summary_df['base_dd'].mean():.1f}%  banda: {summary_df['band_dd'].mean():.1f}%")
    log("\nInterpretazione: se la banda migliora in 4-5 periodi su 5, e' un pattern diffuso,")
    log("non guidato da un singolo periodo anomalo — coerente col principio di robustezza.")
    log("Se migliora solo in 1-2 periodi (anche se il totale aggregato e' positivo), trattare")
    log("con sospetto: il pooling puo' aver amplificato un effetto concentrato, non diffuso.")

    with open("results/calibrate_mr_adx_band.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
