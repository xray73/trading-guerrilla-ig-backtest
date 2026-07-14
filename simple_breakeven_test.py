"""
simple_breakeven_test.py — Ultimo tentativo della sessione (settimo).
Versione DELIBERATAMENTE semplice: stop si sposta UNA VOLTA a breakeven
quando il prezzo raggiunge breakeven_r × R, poi resta fermo lì. NESSUN
tetto (take_profit rimosso, il trade può correre libero fino a scadenza
24h). NESSUN filtro di conferma (primo tocco del nuovo stop chiude il
trade, esattamente come lo stop originale oggi).

Motivazione della semplicità: dato il pattern osservato oggi (6 tentativi
via via più complessi, tutti bocciati in test out-of-sample), si isola
la versione più semplice possibile dell'idea prima di aggiungere
ulteriori gradi di libertà (conferma, filtro velocità) che aumenterebbero
il rischio di overfitting cumulativo sull'intera sessione.

Griglia: breakeven_r ∈ {0.5, 1.0, 1.5} — 3 configurazioni + baseline.
Metrica: PnL / |max_drawdown%| (stessa di staircase_exit_test.py).
Promozione: +10% sul rapporto in train (stesso margine già fissato).
Nessun pavimento sul numero di trade (scelta esplicita dell'utente).

Disciplina walk-forward identica al resto della sessione: train (2023)
→ test (2024-2025) → conferma (3 periodi restanti).
"""

from __future__ import annotations

import dataclasses
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

import engine as eng

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CHUNK_SIZE = 5000
CAPITAL0 = 900.0

BREAKEVEN_LEVELS = [0.5, 1.0, 1.5]
OFFSET_R = 0.1
PROMOTION_MARGIN = 0.10

PERIODS = {
    "2015-2016": ("2015-01-01", "2016-12-31"),
    "2020-covid": ("2020-01-01", "2020-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024-2025": ("2024-01-01", "2025-12-31"),
    "2026-ytd": ("2026-01-01", "2026-07-12"),
}
TRAIN_PERIOD = "2023"
TEST_PERIOD = "2024-2025"
CONFIRM_PERIODS = ["2015-2016", "2020-covid", "2026-ytd"]
WARMUP_DAYS = 90


def d1_query(sql: str, account_id: str, token: str) -> list[dict]:
    url = f"{API_BASE}/{account_id}/d1/database/{DATABASE_ID}/query"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"sql": sql}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"D1 query fallita: {data.get('errors')}")
    return data["result"][0]["results"]


def fetch_all_ohlc(symbol: str, account_id: str, token: str) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        sql = (
            f"SELECT timestamp, open, high, low, close FROM ohlc_prices "
            f"WHERE symbol='{symbol}' ORDER BY timestamp LIMIT {CHUNK_SIZE} OFFSET {offset}"
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
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def slice_period(df, period_label):
    start_str, end_str = PERIODS[period_label]
    start = pd.Timestamp(start_str, tz="UTC") - pd.Timedelta(days=WARMUP_DAYS)
    end = pd.Timestamp(end_str, tz="UTC") + pd.Timedelta(days=1)
    window = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
    return window, pd.Timestamp(start_str, tz="UTC")


def trim_warmup(df, period_start):
    return df[df["timestamp"] >= period_start].reset_index(drop=True)


@dataclasses.dataclass
class BEParams:
    mode: str  # "baseline" | "breakeven"
    breakeven_r: float = 1.0
    offset_r: float = OFFSET_R

    def label(self) -> str:
        return "baseline" if self.mode == "baseline" else f"be_simple_r{self.breakeven_r}"


class SimpleBreakevenEngine(eng.BacktestEngine):
    def __init__(self, capital0, params: BEParams, p=eng.PARAMS, instruments=eng.INSTRUMENTS):
        super().__init__(capital0=capital0, p=p, instruments=instruments)
        self.params = params
        self._pos_state: dict[int, dict] = {}

    def _open_position(self, instrument, direction, bar, atr_at_entry, adx_at_entry):
        n_before = len(self.open_positions)
        super()._open_position(instrument, direction, bar, atr_at_entry, adx_at_entry)
        if len(self.open_positions) > n_before:
            pos = self.open_positions[-1]
            r_unit = abs(pos.entry_price - pos.stop_loss)
            self._pos_state[id(pos)] = {"triggered": False, "r_unit": r_unit}
            if self.params.mode == "breakeven":
                # nessun tetto: take_profit spinto a un valore irraggiungibile
                pos.take_profit = float("inf") if pos.direction == "long" else float("-inf")

    def _try_close_position(self, pos, bar, bar_index, inst):
        if self.params.mode == "breakeven":
            self._update_stop(pos, bar)
        closed = super()._try_close_position(pos, bar, bar_index, inst)
        if closed:
            self._pos_state.pop(id(pos), None)
        return closed

    def _update_stop(self, pos, bar):
        state = self._pos_state.get(id(pos))
        if state is None or state["triggered"]:
            return
        r_unit = state["r_unit"]
        if r_unit <= 0:
            return

        if pos.direction == "long":
            favorable = bar["high"] - pos.entry_price
        else:
            favorable = pos.entry_price - bar["low"]

        if favorable >= self.params.breakeven_r * r_unit:
            offset = self.params.offset_r * r_unit
            if pos.direction == "long":
                pos.stop_loss = pos.entry_price + offset
            else:
                pos.stop_loss = pos.entry_price - offset
            state["triggered"] = True


def eval_config(params: BEParams, period_label: str, full_data: dict) -> dict:
    data = {}
    for name, full_df in full_data.items():
        inst = eng.INSTRUMENTS[name]
        window, period_start = slice_period(full_df, period_label)
        sig = eng.generate_signals(window, inst)
        sig = trim_warmup(sig, period_start)
        data[name] = sig

    engine_ = SimpleBreakevenEngine(capital0=CAPITAL0, params=params)
    trades_df, metrics_df = engine_.run(data)

    num_trades = int(metrics_df["num_trades"].iloc[0])
    pnl_total = float(metrics_df["pnl_total"].iloc[0])
    max_dd = float(metrics_df["max_drawdown_pct"].iloc[0]) if num_trades else np.nan
    risk_adj = pnl_total / abs(max_dd) if (num_trades and not np.isnan(max_dd) and max_dd != 0) else np.nan

    return {
        "config": params.label(),
        "mode": params.mode,
        "breakeven_r": params.breakeven_r if params.mode != "baseline" else None,
        "period": period_label,
        "num_trades": num_trades,
        "win_rate": float(metrics_df["win_rate"].iloc[0]) if num_trades else 0.0,
        "pnl_total": pnl_total,
        "profit_factor": float(metrics_df["profit_factor"].iloc[0]) if num_trades else np.nan,
        "max_drawdown_pct": max_dd,
        "risk_adj_pnl": risk_adj,
    }


def build_grid():
    return [BEParams(mode="baseline")] + [BEParams(mode="breakeven", breakeven_r=r) for r in BREAKEVEN_LEVELS]


def passes(row, baseline):
    if row["num_trades"] == 0 or np.isnan(row["risk_adj_pnl"]) or np.isnan(baseline["risk_adj_pnl"]):
        return False
    return row["risk_adj_pnl"] >= baseline["risk_adj_pnl"] * (1 + PROMOTION_MARGIN)


def main():
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        print("ERRORE: secrets mancanti.", file=sys.stderr)
        sys.exit(1)

    os.makedirs("results", exist_ok=True)
    print("Scarico OHLC DAX + FTSE100...")
    full_data = {
        "DAX": fetch_all_ohlc("DAX", account_id, token),
        "FTSE100": fetch_all_ohlc("FTSE100", account_id, token),
    }

    grid = build_grid()
    all_rows = []

    print(f"\n=== TRAIN ({TRAIN_PERIOD}) ===")
    train_rows = [eval_config(p, TRAIN_PERIOD, full_data) for p in grid]
    for r in train_rows:
        print(f"  {r['config']:16s} trades={r['num_trades']:4d} win_rate={r['win_rate']*100:5.2f}% "
              f"pnl={r['pnl_total']:8.0f} dd={r['max_drawdown_pct']*100:6.2f}% pnl/dd={r['risk_adj_pnl']:.1f}")
    all_rows += train_rows
    pd.DataFrame(all_rows).to_csv("results/simple_be_train.csv", index=False)

    baseline_train = train_rows[0]
    candidates = [r for r in train_rows[1:] if passes(r, baseline_train)]
    if not candidates:
        print("\nNessuna configurazione promossa. Chiuso qui.")
        return

    print(f"\nPromosse: {[c['config'] for c in candidates]}")
    grid_by_label = {p.label(): p for p in grid}

    print(f"\n=== TEST ({TEST_PERIOD}) ===")
    test_rows = [eval_config(BEParams(mode="baseline"), TEST_PERIOD, full_data)]
    print(f"  baseline: pnl/dd={test_rows[0]['risk_adj_pnl']:.1f} pnl={test_rows[0]['pnl_total']:.0f}")
    survivors = []
    for c in candidates:
        row = eval_config(grid_by_label[c["config"]], TEST_PERIOD, full_data)
        test_rows.append(row)
        ok = passes(row, test_rows[0])
        print(f"  {row['config']:16s} pnl/dd={row['risk_adj_pnl']:.1f} pnl={row['pnl_total']:8.0f}  "
              f"{'SUPERA' if ok else 'non supera'}")
        if ok:
            survivors.append(grid_by_label[c["config"]])
    all_rows += test_rows
    pd.DataFrame(all_rows).to_csv("results/simple_be_train_test.csv", index=False)

    if not survivors:
        print("\nNessuna sopravvive al test. Chiuso qui.")
        return

    print(f"\n=== CONFERMA ===")
    confirm_rows = []
    for period in CONFIRM_PERIODS:
        b = eval_config(BEParams(mode="baseline"), period, full_data)
        confirm_rows.append(b)
        print(f"  [{period}] baseline: pnl/dd={b['risk_adj_pnl']:.1f} pnl={b['pnl_total']:.0f}")
        for p in survivors:
            row = eval_config(p, period, full_data)
            confirm_rows.append(row)
            print(f"  [{period}] {row['config']:16s} pnl/dd={row['risk_adj_pnl']:.1f} pnl={row['pnl_total']:8.0f}")
    all_rows += confirm_rows
    pd.DataFrame(all_rows).to_csv("results/simple_be_full.csv", index=False)
    print("\nCompletato.")


if __name__ == "__main__":
    main()
