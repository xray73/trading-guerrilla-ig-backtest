"""
plot_trade_windows.py — Genera grafici candela+EMA+trade reali per revisione
visiva, uno per (periodo validato x strumento), scegliendo AUTOMATICAMENTE
la finestra di ~3 settimane con la maggior concentrazione di perdite
"falso segnale" — nessuna selezione manuale, per evitare hindsight bias
nella scelta di quali finestre guardare.

Richiede due secrets del repository GitHub (stessi già configurati per
ema_grid.yml):
  CLOUDFLARE_API_TOKEN   (permesso "D1 read" o superiore)
  CLOUDFLARE_ACCOUNT_ID

Output: un PNG per (periodo, strumento) in results/, es.
  DAX_2023_window.png, FTSE100_2023_window.png, ...

Motore/segnale di riferimento: run_id 7-11 in backtest_runs (Variante 6,
motore v2, già validati — Charter sez.3, RCA sez.11). Non modifica né
rilancia il motore, legge solo trades già salvati in D1.
"""

from __future__ import annotations

import os
import sys
import json
import time

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CHUNK_SIZE = 5000

INSTRUMENTS = ["DAX", "FTSE100"]

# run_id, periodo, confini periodo (per clip finestra ai limiti reali)
RUNS = [
    (7,  "2015-2016",  "2015-01-01", "2016-12-31"),
    (8,  "2020-covid", "2020-01-01", "2020-12-31"),
    (9,  "2023",       "2023-01-01", "2023-12-31"),
    (10, "2024-2025",  "2024-01-01", "2025-12-31"),
    (11, "2026-ytd",   "2026-01-01", "2026-07-12"),
]

WINDOW_DAYS = 21       # ampiezza finestra di visualizzazione (~3 settimane)
BUCKET_DAYS = 7        # granularità di ricerca del cluster di perdite
CONTEXT_BEFORE_DAYS = 3  # margine prima del cluster, per vedere il contesto


def d1_query(sql: str, account_id: str, token: str) -> list[dict]:
    url = f"{API_BASE}/{account_id}/d1/database/{DATABASE_ID}/query"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"sql": sql}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"D1 query fallita: {data.get('errors')}")
    return data["result"][0]["results"]


def fetch_ohlc_range(symbol: str, start: str, end: str, account_id: str, token: str) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        sql = (
            f"SELECT timestamp, open, high, low, close FROM ohlc_prices "
            f"WHERE symbol='{symbol}' AND timestamp >= '{start}' AND timestamp < '{end}' "
            f"ORDER BY timestamp LIMIT {CHUNK_SIZE} OFFSET {offset}"
        )
        batch = d1_query(sql, account_id, token)
        if not batch:
            break
        rows.extend(batch)
        offset += CHUNK_SIZE
        if len(batch) < CHUNK_SIZE:
            break
        time.sleep(0.15)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_trades(run_id: int, symbol: str, account_id: str, token: str) -> pd.DataFrame:
    sql = (
        f"SELECT direction, entry_time, entry_price, exit_time, exit_price, "
        f"pnl, causa_esito FROM trades WHERE run_id={run_id} AND symbol='{symbol}' "
        f"ORDER BY entry_time"
    )
    rows = d1_query(sql, account_id, token)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return df


def find_worst_window(trades: pd.DataFrame, period_start: pd.Timestamp,
                       period_end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Trova la finestra di WINDOW_DAYS con più perdite 'falso segnale',
    cercando su bucket di BUCKET_DAYS lungo tutto il periodo. Nessuna
    selezione manuale: puramente meccanico, stessa regola per tutti."""
    losses = trades[trades["causa_esito"] == "falso segnale"]
    if losses.empty:
        # fallback: nessuna perdita di questo tipo, usa le prime 3 settimane
        return period_start, period_start + pd.Timedelta(days=WINDOW_DAYS)

    bucket_start = period_start
    best_start, best_count = bucket_start, -1
    while bucket_start < period_end:
        bucket_end = bucket_start + pd.Timedelta(days=BUCKET_DAYS)
        count = ((losses["entry_time"] >= bucket_start) &
                  (losses["entry_time"] < bucket_end)).sum()
        if count > best_count:
            best_count = count
            best_start = bucket_start
        bucket_start += pd.Timedelta(days=BUCKET_DAYS)

    window_start = max(period_start, best_start - pd.Timedelta(days=CONTEXT_BEFORE_DAYS))
    window_end = min(period_end, window_start + pd.Timedelta(days=WINDOW_DAYS))
    return window_start, window_end


def build_chart(symbol: str, period_label: str, ohlc: pd.DataFrame,
                 trades_window: pd.DataFrame, window_start: pd.Timestamp,
                 output_path: str):
    df = ohlc.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    view = df[df["timestamp"] >= window_start].reset_index(drop=True)
    if view.empty:
        print(f"  ATTENZIONE {symbol}/{period_label}: nessuna barra nella finestra, salto.")
        return

    fig, ax = plt.subplots(figsize=(22, 10), dpi=130)
    width = pd.Timedelta(minutes=20)

    for _, row in view.iterrows():
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax.plot([row["timestamp"], row["timestamp"]], [row["low"], row["high"]],
                color=color, linewidth=0.8, zorder=2)
        body_low = min(row["open"], row["close"])
        body_high = max(row["open"], row["close"])
        ax.add_patch(Rectangle(
            (mdates.date2num(row["timestamp"]) - width.total_seconds() / 86400 / 2, body_low),
            width.total_seconds() / 86400, max(body_high - body_low, 0.5),
            facecolor=color, edgecolor=color, zorder=3))

    ax.plot(view["timestamp"], view["ema20"], color="#42a5f5", linewidth=1.1, label="EMA20")
    ax.plot(view["timestamp"], view["ema50"], color="#ff9800", linewidth=1.1, label="EMA50")
    ax.plot(view["timestamp"], view["ema100"], color="#ab47bc", linewidth=1.4, label="EMA100")
    ax.plot(view["timestamp"], view["ema200"], color="#8d6e63", linewidth=1.6, label="EMA200")

    for _, t in trades_window.iterrows():
        win = t["pnl"] > 0
        marker_color = "#00c853" if win else "#d50000"
        marker = "^" if t["direction"] == "long" else "v"
        ax.scatter(t["entry_time"], t["entry_price"], marker=marker, s=130,
                   color=marker_color, edgecolor="black", linewidth=0.6, zorder=5)
        ax.scatter(t["exit_time"], t["exit_price"], marker="x", s=70,
                   color=marker_color, linewidth=2, zorder=5)
        ax.plot([t["entry_time"], t["exit_time"]], [t["entry_price"], t["exit_price"]],
                color=marker_color, linewidth=0.9, linestyle="--", alpha=0.6, zorder=4)
        if not win:
            label = t["causa_esito"] or "?"
            ax.annotate(label, (t["exit_time"], t["exit_price"]), fontsize=7,
                        color=marker_color, xytext=(3, -10), textcoords="offset points")

    ax.set_title(
        f"{symbol} 30min — {period_label} — finestra auto-selezionata "
        f"(max densità 'falso segnale') — EMA20/50/100/200 + trade reali\n"
        f"▲/▼ = ingresso long/short   × = uscita   verde = vincente   rosso = perdente",
        fontsize=11)
    ax.set_ylabel(f"Prezzo {symbol}")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close(fig)


def main():
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        print("ERRORE: CLOUDFLARE_API_TOKEN o CLOUDFLARE_ACCOUNT_ID mancanti.",
              file=sys.stderr)
        sys.exit(1)

    os.makedirs("results", exist_ok=True)
    summary = []

    for run_id, period_label, p_start, p_end in RUNS:
        period_start = pd.Timestamp(p_start, tz="UTC")
        period_end = pd.Timestamp(p_end, tz="UTC")

        for symbol in INSTRUMENTS:
            print(f"\n=== {symbol} / {period_label} (run_id={run_id}) ===")
            trades = fetch_trades(run_id, symbol, account_id, token)
            if trades.empty:
                print("  Nessun trade trovato, salto.")
                continue

            window_start, window_end = find_worst_window(trades, period_start, period_end)
            print(f"  Finestra selezionata: {window_start.date()} -> {window_end.date()}")

            # warmup per EMA200: quanto disponibile prima della finestra,
            # senza uscire dai confini del periodo (stessa limitazione del
            # motore reale a inizio periodo — nessun dato storico esterno)
            warmup_start = max(period_start, window_start - pd.Timedelta(days=60))
            ohlc = fetch_ohlc_range(symbol, warmup_start.strftime("%Y-%m-%d"),
                                     window_end.strftime("%Y-%m-%d"), account_id, token)
            if ohlc.empty:
                print("  Nessun dato OHLC trovato, salto.")
                continue

            trades_window = trades[
                (trades["entry_time"] >= window_start) & (trades["entry_time"] < window_end)
            ]
            n_losses = (trades_window["causa_esito"] == "falso segnale").sum()

            out_path = f"results/{symbol}_{period_label.replace(' ', '_')}_window.png"
            build_chart(symbol, period_label, ohlc, trades_window, window_start, out_path)
            print(f"  Salvato {out_path} ({len(trades_window)} trade, "
                  f"{n_losses} perdite 'falso segnale' nella finestra)")

            summary.append({
                "symbol": symbol, "period": period_label,
                "window_start": str(window_start.date()),
                "window_end": str(window_end.date()),
                "n_trades_in_window": int(len(trades_window)),
                "n_falso_segnale_in_window": int(n_losses),
                "file": out_path,
            })

    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== RIEPILOGO ===")
    for s in summary:
        print(f"{s['symbol']:10s} {s['period']:12s} {s['window_start']}..{s['window_end']}  "
              f"trade={s['n_trades_in_window']:3d}  falso_segnale={s['n_falso_segnale_in_window']:3d}")


if __name__ == "__main__":
    main()
