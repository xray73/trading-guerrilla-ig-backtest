"""
check_timeframe_comparison.py — CONTO GREZZO (non un test causale
completo): confronta V6 su 30min (baseline) vs 1h/2h/4h, sui 5 periodi
ufficiali, usando i dati OHLC 30min gia' in D1 (nessun nuovo fetch
Dukascopy — resample locale).

REGOLA FISSATA PRIMA DI VEDERE I RISULTATI:
- NESSUN parametro ricalibrato per timeframe. EMA20/50, ADX14, breakout
  lookback 20/40, ATR14, max_holding_bars=48 — tutti INVARIATI in
  numero di barre. Cambia solo la granularita' delle barre in input.
  (Ricalibrare per ogni timeframe introdurrebbe un grado di liberta'
  in piu' — non staremmo piu' isolando l'effetto "meno rumore".)
- Resample OHLC standard: open=primo, high=max, low=min, close=ultimo
  della finestra.
- Stesso capitale (1400 EUR, pool V6), stesso motore
  (BacktestEngineFloatingKillSwitch), stessi 5 periodi ufficiali.
- Segnali generati sulla serie COMPLETA resample-ata (2015-2026) per
  garantire warmup corretto degli indicatori, poi filtrati per periodo
  prima di passarli al motore (un'istanza nuova per ogni combinazione
  timeframe x periodo, capital0=1400 sempre).

Output: SOLO tabella aggregata (trade/PnL/winrate/PF per timeframe x
periodo + totale), nessun dato individuale.
"""
import os
import pandas as pd

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from ohlc_data_source import get_ohlc

CF_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
CF_API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]

CAPITAL_V6 = 1400.0
TIMEFRAMES = {"30min": None, "1h": "1h", "2h": "2h", "4h": "4h"}

PERIODS = {
    "2015-2016": ("2015-01-01", "2016-12-31"),
    "2020-covid": ("2020-01-01", "2020-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024-2025": ("2024-01-01", "2025-12-31"),
    "2026-ytd": ("2026-01-01", "2026-07-12"),
}


def resample_ohlc(df: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    if rule is None:
        return df
    r = (df.set_index("timestamp")
           .resample(rule, label="left", closed="left")
           .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
           .dropna()
           .reset_index())
    return r


def slice_period(signals: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return signals[(signals["timestamp"] >= start_ts) & (signals["timestamp"] < end_ts)].reset_index(drop=True)


def run_period(signals_by_instrument: dict, start: str, end: str) -> dict:
    sliced = {name: slice_period(sig, start, end) for name, sig in signals_by_instrument.items()}
    engine_ = BacktestEngineFloatingKillSwitch(capital0=CAPITAL_V6)
    trades_df, metrics_df = engine_.run(sliced)
    n = len(trades_df)
    pnl = float(trades_df["pnl"].sum()) if n else 0.0
    wins = trades_df[trades_df["pnl"] > 0]["pnl"].sum() if n else 0.0
    losses = -trades_df[trades_df["pnl"] <= 0]["pnl"].sum() if n else 0.0
    pf = (wins / losses) if losses > 0 else float("nan")
    win_rate = (trades_df["pnl"] > 0).mean() if n else float("nan")
    return {"n_trades": n, "pnl": pnl, "win_rate": win_rate, "pf": pf}


def main():
    print("Scarico OHLC 30min continuo (DAX+FTSE100), gia' in D1...")
    hist30 = {}
    for name in ("DAX", "FTSE100"):
        hist30[name] = get_ohlc(name, CF_ACCOUNT_ID, CF_API_TOKEN)
        print(f"  {name}: {len(hist30[name])} barre 30min")

    results = []

    for tf_label, tf_rule in TIMEFRAMES.items():
        print(f"\n=== Timeframe {tf_label} ===")
        hist_tf = {name: resample_ohlc(hist30[name], tf_rule) for name in hist30}
        for name in hist_tf:
            print(f"  {name}: {len(hist_tf[name])} barre dopo resample")

        signals_tf = {name: eng.generate_signals(hist_tf[name], eng.INSTRUMENTS[name]) for name in hist_tf}

        period_results = {}
        for period_name, (start, end) in PERIODS.items():
            r = run_period(signals_tf, start, end)
            period_results[period_name] = r
            print(f"  [{period_name}] trade={r['n_trades']} pnl={r['pnl']:+.2f} "
                  f"win_rate={r['win_rate']:.1%} pf={r['pf']:.3f}")

        total_trades = sum(r["n_trades"] for r in period_results.values())
        total_pnl = sum(r["pnl"] for r in period_results.values())
        total_wins = sum(r["pnl"] for r in period_results.values() if r["pnl"] > 0)  # approssimato sotto
        n_periods_positive = sum(1 for r in period_results.values() if r["pnl"] > 0)

        results.append({"timeframe": tf_label, "trade_totali": total_trades,
                         "pnl_totale": total_pnl, "periodi_positivi": f"{n_periods_positive}/5"})

    print("\n=== TABELLA RIASSUNTIVA (conto grezzo, non ancora bootstrap) ===")
    print(f"{'Timeframe':<10}{'Trade tot':>12}{'PnL totale':>14}{'Periodi +':>12}")
    for r in results:
        print(f"{r['timeframe']:<10}{r['trade_totali']:>12}{r['pnl_totale']:>14.2f}{r['periodi_positivi']:>12}")


if __name__ == "__main__":
    main()
