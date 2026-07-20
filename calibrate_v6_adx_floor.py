"""
calibrate_v6_adx_floor.py — Verifica se alzare la soglia minima ADX
per l'ingresso V6 (attualmente >20) a 25 o 30 migliora il risultato,
dopo che il bucket analysis (19/07/2026) ha mostrato che la fascia
20-25 (37% del volume V6) ha l'expectancy più bassa di tutte (+6,97€
contro il picco +15,97€ a 30-35).

STESSA NOTA METODOLOGICA del test gemello per MR
(calibrate_mr_adx_band.py): la soglia candidata nasce dal pooling di
tutti i 5 periodi insieme, quindi non c'e' un vero fuori-campione —
si verifica PERIODO PER PERIODO se il vantaggio e' diffuso o
concentrato (principio 3 Protocollo Anti-Rumore), non un train/test
classico.

ASPETTATIVA DICHIARATA PRIMA DI VEDERE I RISULTATI (per onestà, non
per pilotare l'esito): il test gemello su MR ha mostrato un risultato
peggiore del previsto per via della path-dependency del capitale
(rimuovere trade cambia quali trade successivi sono eleggibili).
Per V6 questo canale specifico e' assente (V6 forza sempre al minimo,
non salta mai) ma la fascia 20-25 rimossa e' comunque profittevole
(non in perdita come la fascia MR rimossa) — aspettativa dichiarata:
risultato probabilmente negativo o piatto, non positivo.

METODO: stesso motore BacktestEngineFloatingKillSwitch INVARIATO — la
soglia e' applicata a livello di segnale (pre-filtro), non una
modifica al motore.

Dati letti da D1 via ohlc_data_source.py. Nessuna scrittura su D1
oltre l'aggiornamento incrementale. Nessuna modifica a engine.py o
engine_floating_kill_switch.py.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from ohlc_data_source import get_ohlc

CAPITAL0 = 2000.0
SYMBOLS = ["DAX", "FTSE100"]
CANDIDATE_THRESHOLDS = [25.0, 30.0]

PERIODS = [
    ("2015-2016", "2015-01-05", "2016-12-29"),
    ("2020-covid", "2020-01-02", "2020-12-30"),
    ("2023", "2023-01-02", "2023-12-30"),
    ("2024-2025", "2024-01-03", "2025-12-31"),
    ("2026-ytd", "2026-01-05", "2026-07-10"),
]


def slice_period(df: pd.DataFrame, p_start: pd.Timestamp, p_end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["timestamp"] >= p_start) & (df["timestamp"] < p_end)].reset_index(drop=True)


def apply_adx_floor(signals: pd.DataFrame, adx_min: float) -> pd.DataFrame:
    out = signals.copy()
    out.loc[out["adx"] < adx_min, "signal"] = None
    return out


def metrics_summary(trades_df: pd.DataFrame, capital0: float) -> dict:
    if trades_df.empty:
        return {"n_trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan,
                "pnl_total": 0.0, "max_drawdown_pct": 0.0}
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    sum_wins, sum_losses = wins["pnl"].sum(), losses["pnl"].sum()
    pf = sum_wins / abs(sum_losses) if sum_losses != 0 else np.inf
    equity = capital0 + trades_df["pnl"].cumsum()
    running_max = equity.cummax()
    dd = ((equity - running_max) / running_max).min() * 100
    return {
        "n_trades": len(trades_df), "win_rate_pct": 100 * len(wins) / len(trades_df),
        "profit_factor": pf, "pnl_total": trades_df["pnl"].sum(), "max_drawdown_pct": dd,
    }


def main():
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=== Verifica soglia ADX minima V6 (25/30 vs 20 attuale) — periodo per periodo ===\n")
    log("Aspettativa dichiarata prima dei risultati: probabilmente negativo o piatto, "
        "coerente col test gemello MR (path-dependency + fascia rimossa comunque profittevole).\n")

    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        log("ERRORE: CLOUDFLARE_API_TOKEN o CLOUDFLARE_ACCOUNT_ID mancanti.")
        return

    log("Verifico/aggiorno OHLC (D1 + eventuali barre mancanti da Dukascopy)...")
    raw_full = {name: get_ohlc(name, account_id, token, log=log) for name in SYMBOLS}
    log("Fatto.\n")

    v6_signals_full = {name: eng.generate_signals(raw_full[name], eng.INSTRUMENTS[name]) for name in SYMBOLS}
    v6_signals_by_threshold = {
        thr: {name: apply_adx_floor(v6_signals_full[name], thr) for name in SYMBOLS}
        for thr in CANDIDATE_THRESHOLDS
    }

    rows = []
    for label, p_start_str, p_end_str in PERIODS:
        p_start = pd.Timestamp(p_start_str, tz="UTC")
        p_end = pd.Timestamp(p_end_str, tz="UTC") + pd.Timedelta(days=1)
        log(f"Periodo {label}")

        sig_base = {name: slice_period(v6_signals_full[name], p_start, p_end) for name in SYMBOLS}
        eng_base = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
        trades_base, _ = eng_base.run(sig_base)
        m_base = metrics_summary(trades_base, CAPITAL0)
        log(f"  Attuale (ADX>20):  n={m_base['n_trades']:3d}  WR={m_base['win_rate_pct']:.1f}%  "
            f"PF={m_base['profit_factor']:.2f}  PnL={m_base['pnl_total']:+.2f}  maxDD={m_base['max_drawdown_pct']:.1f}%")

        row = {"periodo": label, "base_n": m_base["n_trades"], "base_pnl": m_base["pnl_total"],
               "base_dd": m_base["max_drawdown_pct"]}

        for thr in CANDIDATE_THRESHOLDS:
            sig_thr = {name: slice_period(v6_signals_by_threshold[thr][name], p_start, p_end) for name in SYMBOLS}
            eng_thr = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0)
            trades_thr, _ = eng_thr.run(sig_thr)
            m_thr = metrics_summary(trades_thr, CAPITAL0)
            delta = m_thr["pnl_total"] - m_base["pnl_total"]
            log(f"  ADX>{thr:.0f}:          n={m_thr['n_trades']:3d}  WR={m_thr['win_rate_pct']:.1f}%  "
                f"PF={m_thr['profit_factor']:.2f}  PnL={m_thr['pnl_total']:+.2f}  maxDD={m_thr['max_drawdown_pct']:.1f}%  "
                f"(delta: {delta:+.2f})")
            row[f"thr{thr:.0f}_n"] = m_thr["n_trades"]
            row[f"thr{thr:.0f}_pnl"] = m_thr["pnl_total"]
            row[f"thr{thr:.0f}_dd"] = m_thr["max_drawdown_pct"]
            row[f"thr{thr:.0f}_delta"] = delta
        log("")
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv("results/calibrate_v6_adx_floor.csv", index=False)

    log(f"{'='*70}\nRIEPILOGO\n{'='*70}")
    log(f"PnL totale — attuale (ADX>20): {summary_df['base_pnl'].sum():+.2f}")
    for thr in CANDIDATE_THRESHOLDS:
        col = f"thr{thr:.0f}"
        log(f"ADX>{thr:.0f}: PnL={summary_df[col+'_pnl'].sum():+.2f}  "
            f"delta={summary_df[col+'_delta'].sum():+.2f}  "
            f"periodi migliori={int((summary_df[col+'_delta']>0).sum())}/5  "
            f"DD medio={summary_df[col+'_dd'].mean():.1f}%")

    log("\nInterpretazione: se una soglia migliora in 4-5 periodi su 5, e' un pattern diffuso.")
    log("Se migliora solo in 1-2 periodi (anche con aggregato positivo), trattare con sospetto.")

    with open("results/calibrate_v6_adx_floor.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
