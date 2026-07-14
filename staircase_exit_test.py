"""
staircase_exit_test.py — Testa un'uscita "a scaletta" su multipli di R,
diversa sia dal baseline (target fisso 2R) sia dal trailing continuo già
testato (exit_variants_test.py, bocciato). Qui i livelli sono discreti e
il tetto è molto più alto (fino a 5R), dando spazio reale ai trade che
continuano a correre — non solo protezione tra 0 e 2R.

Meccanismo (esempio con breakeven=1.0R, lockin=2.0R, cap=4.0R):
  - Sotto 1R di favore:  stop originale (ATR x 1.5) — identico a oggi
  - Raggiunto 1R:        stop sale a breakeven + piccolo offset (0.1R,
                          copre lo spread, garantisce vittoria vera)
  - Superato 2R:         stop sale a 2R (se inverte da qui, esce con
                          almeno +2R garantito — mai più un trade da
                          "quasi vinto" a perdita)
  - Raggiunto 4R:        CHIUSURA IMMEDIATA — nuovo tetto, il trade non
                          continua oltre indipendentemente da cosa fa il
                          prezzo dopo

Il "tetto" è implementato come nuovo take_profit fissato all'apertura
(non un continuo trailing) — tecnicamente più semplice e prevedibile del
trailing ATR continuo già bocciato, e lascia spazio reale fino al tetto
invece di restare quasi sempre ancorato a 2R come nella versione prima.

Metrica di selezione: PnL_totale / |max_drawdown%| (rapporto tipo
Calmar) — non più win_rate/profit_factor, per esplicita richiesta
dell'utente dopo aver chiarito l'obiettivo reale (profitto corretto per
il rischio, non selettività del segnale).

Criterio di promozione (fissato PRIMA di vedere risultati): il rapporto
PnL/drawdown del candidato deve superare quello del baseline di almeno
+10% in train per essere promosso al test out-of-sample. Nessun
pavimento sul numero di trade — riportato solo come contesto, non come
criterio di scarto (scelta esplicita dell'utente).

Disciplina walk-forward identica al resto della sessione: train (2023)
→ test (2024-2025, nessun ritocco) → conferma (3 periodi restanti, solo
riportata).
"""

from __future__ import annotations

import dataclasses
import itertools
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
LOCKIN_LEVELS = [2.0, 3.0]
CAP_LEVELS = [3.0, 4.0, 5.0]
OFFSET_R = 0.1  # fisso, copre lo spread con margine

PROMOTION_MARGIN = 0.10  # +10% sul rapporto PnL/drawdown per promuovere al test

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


# =====================================================================
# 1. FETCH DATI
# =====================================================================

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


def slice_period(df: pd.DataFrame, period_label: str) -> tuple[pd.DataFrame, pd.Timestamp]:
    start_str, end_str = PERIODS[period_label]
    start = pd.Timestamp(start_str, tz="UTC") - pd.Timedelta(days=WARMUP_DAYS)
    end = pd.Timestamp(end_str, tz="UTC") + pd.Timedelta(days=1)
    window = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
    return window, pd.Timestamp(start_str, tz="UTC")


def trim_warmup(df: pd.DataFrame, period_start: pd.Timestamp) -> pd.DataFrame:
    return df[df["timestamp"] >= period_start].reset_index(drop=True)


# =====================================================================
# 2. MOTORE ESTESO — uscita a scaletta
# =====================================================================

@dataclasses.dataclass
class StaircaseParams:
    mode: str  # "baseline" | "staircase"
    breakeven_r: float = 1.0
    lockin_r: float = 2.0
    cap_r: float = 4.0
    offset_r: float = OFFSET_R

    def label(self) -> str:
        if self.mode == "baseline":
            return "baseline"
        return f"stair_be{self.breakeven_r}_lock{self.lockin_r}_cap{self.cap_r}"


class StaircaseEngine(eng.BacktestEngine):
    def __init__(self, capital0: float, params: StaircaseParams,
                 p: eng.ChartaParams = eng.PARAMS,
                 instruments: dict = eng.INSTRUMENTS):
        super().__init__(capital0=capital0, p=p, instruments=instruments)
        self.params = params
        self._pos_state: dict[int, dict] = {}

    def _open_position(self, instrument, direction, bar, atr_at_entry, adx_at_entry):
        n_before = len(self.open_positions)
        super()._open_position(instrument, direction, bar, atr_at_entry, adx_at_entry)
        if len(self.open_positions) > n_before:
            pos = self.open_positions[-1]
            r_unit = abs(pos.entry_price - pos.stop_loss)
            self._pos_state[id(pos)] = {"extreme_price": pos.entry_price, "stage": 0, "r_unit": r_unit}

            if self.params.mode == "staircase" and r_unit > 0:
                cap_dist = self.params.cap_r * r_unit
                if pos.direction == "long":
                    pos.take_profit = pos.entry_price + cap_dist
                else:
                    pos.take_profit = pos.entry_price - cap_dist

    def _try_close_position(self, pos, bar, bar_index, inst):
        if self.params.mode == "staircase":
            self._update_dynamic_stop(pos, bar)
        closed = super()._try_close_position(pos, bar, bar_index, inst)
        if closed:
            self._pos_state.pop(id(pos), None)
        return closed

    def _update_dynamic_stop(self, pos, bar):
        state = self._pos_state.get(id(pos))
        if state is None:
            return
        r_unit = state["r_unit"]
        if r_unit <= 0:
            return

        if pos.direction == "long":
            state["extreme_price"] = max(state["extreme_price"], bar["high"])
            favorable = state["extreme_price"] - pos.entry_price
        else:
            state["extreme_price"] = min(state["extreme_price"], bar["low"])
            favorable = pos.entry_price - state["extreme_price"]

        pr = self.params

        # scaletta: si sale di stage, mai si scende (ratchet)
        if favorable >= pr.lockin_r * r_unit and state["stage"] < 2:
            state["stage"] = 2
            offset = pr.lockin_r * r_unit
            if pos.direction == "long":
                pos.stop_loss = max(pos.stop_loss, pos.entry_price + offset)
            else:
                pos.stop_loss = min(pos.stop_loss, pos.entry_price - offset)
        elif favorable >= pr.breakeven_r * r_unit and state["stage"] < 1:
            state["stage"] = 1
            offset = pr.offset_r * r_unit
            if pos.direction == "long":
                pos.stop_loss = max(pos.stop_loss, pos.entry_price + offset)
            else:
                pos.stop_loss = min(pos.stop_loss, pos.entry_price - offset)


# =====================================================================
# 3. VALUTAZIONE
# =====================================================================

def eval_config(params: StaircaseParams, period_label: str,
                 full_data: dict[str, pd.DataFrame]) -> dict:
    data = {}
    for name, full_df in full_data.items():
        inst = eng.INSTRUMENTS[name]
        window, period_start = slice_period(full_df, period_label)
        sig = eng.generate_signals(window, inst)
        sig = trim_warmup(sig, period_start)
        data[name] = sig

    engine_ = StaircaseEngine(capital0=CAPITAL0, params=params)
    trades_df, metrics_df = engine_.run(data)

    num_trades = int(metrics_df["num_trades"].iloc[0])
    pnl_total = float(metrics_df["pnl_total"].iloc[0])
    max_dd = float(metrics_df["max_drawdown_pct"].iloc[0]) if num_trades else np.nan
    risk_adj = pnl_total / abs(max_dd) if (num_trades and max_dd not in (0, np.nan) and not np.isnan(max_dd)) else np.nan

    return {
        "config": params.label(),
        "mode": params.mode,
        "breakeven_r": params.breakeven_r if params.mode != "baseline" else None,
        "lockin_r": params.lockin_r if params.mode != "baseline" else None,
        "cap_r": params.cap_r if params.mode != "baseline" else None,
        "period": period_label,
        "num_trades": num_trades,
        "win_rate": float(metrics_df["win_rate"].iloc[0]) if num_trades else 0.0,
        "pnl_total": pnl_total,
        "profit_factor": float(metrics_df["profit_factor"].iloc[0]) if num_trades else np.nan,
        "max_drawdown_pct": max_dd,
        "risk_adj_pnl": risk_adj,
    }


def build_grid() -> list[StaircaseParams]:
    grid = [StaircaseParams(mode="baseline")]
    for be, lock, cap in itertools.product(BREAKEVEN_LEVELS, LOCKIN_LEVELS, CAP_LEVELS):
        if be < lock < cap:
            grid.append(StaircaseParams(mode="staircase", breakeven_r=be, lockin_r=lock, cap_r=cap))
    return grid


def passes_criteria(row: dict, baseline_row: dict) -> bool:
    if row["num_trades"] == 0 or np.isnan(row["risk_adj_pnl"]) or np.isnan(baseline_row["risk_adj_pnl"]):
        return False
    return row["risk_adj_pnl"] >= baseline_row["risk_adj_pnl"] * (1 + PROMOTION_MARGIN)


# =====================================================================
# 4. MAIN
# =====================================================================

def main():
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        print("ERRORE: CLOUDFLARE_API_TOKEN o CLOUDFLARE_ACCOUNT_ID mancanti.", file=sys.stderr)
        sys.exit(1)

    os.makedirs("results", exist_ok=True)

    print("Scarico OHLC DAX + FTSE100...")
    full_data = {
        "DAX": fetch_all_ohlc("DAX", account_id, token),
        "FTSE100": fetch_all_ohlc("FTSE100", account_id, token),
    }
    for name, df in full_data.items():
        print(f"  {name}: {len(df)} barre")

    grid = build_grid()
    print(f"\n{len(grid)} configurazioni totali (1 baseline + {len(grid)-1} scalette)")

    all_rows = []

    # --- TRAIN ---
    print(f"\n=== TRAIN ({TRAIN_PERIOD}) ===")
    train_rows = []
    for params in grid:
        row = eval_config(params, TRAIN_PERIOD, full_data)
        train_rows.append(row)
        ra = row['risk_adj_pnl']
        ra_str = f"{ra:.1f}" if not np.isnan(ra) else "n/d"
        print(f"  {row['config']:32s} trades={row['num_trades']:4d} "
              f"win_rate={row['win_rate']*100:5.2f}% pnl={row['pnl_total']:8.0f} "
              f"dd={row['max_drawdown_pct']*100:6.2f}% pnl/dd={ra_str}")
    all_rows += train_rows

    baseline_train = next(r for r in train_rows if r["mode"] == "baseline")
    print(f"\n  Baseline train: pnl/dd={baseline_train['risk_adj_pnl']:.2f}")
    print(f"  Soglia promozione (+{PROMOTION_MARGIN*100:.0f}%): "
          f"{baseline_train['risk_adj_pnl']*(1+PROMOTION_MARGIN):.2f}")

    candidates = [r for r in train_rows if r["mode"] != "baseline" and passes_criteria(r, baseline_train)]
    pd.DataFrame(all_rows).to_csv("results/staircase_train.csv", index=False)

    if not candidates:
        print("\n  NESSUNA configurazione promossa al test. Chiuso qui.")
        return

    candidates.sort(key=lambda r: r["risk_adj_pnl"], reverse=True)
    print(f"\n  PROMOSSE AL TEST ({len(candidates)}): " + ", ".join(c["config"] for c in candidates))

    # --- TEST ---
    print(f"\n=== TEST ({TEST_PERIOD}, out-of-sample) ===")
    test_rows = []
    baseline_test = eval_config(StaircaseParams(mode="baseline"), TEST_PERIOD, full_data)
    test_rows.append(baseline_test)
    print(f"  baseline: pnl/dd={baseline_test['risk_adj_pnl']:.2f} pnl={baseline_test['pnl_total']:.0f}")

    grid_by_label = {p.label(): p for p in grid}
    survivors = []
    for c in candidates:
        params = grid_by_label[c["config"]]
        row = eval_config(params, TEST_PERIOD, full_data)
        test_rows.append(row)
        passed = passes_criteria(row, baseline_test)
        print(f"  {row['config']:32s} pnl/dd={row['risk_adj_pnl']:.2f} pnl={row['pnl_total']:8.0f}  "
              f"{'SUPERA' if passed else 'non supera'}")
        if passed:
            survivors.append(params)

    all_rows += test_rows
    pd.DataFrame(all_rows).to_csv("results/staircase_train_test.csv", index=False)

    if not survivors:
        print("\n  NESSUNA configurazione supera il test out-of-sample. Chiuso qui.")
        return

    print(f"\n  SOPRAVVISSUTE ({len(survivors)}): " + ", ".join(s.label() for s in survivors))

    # --- CONFIRM ---
    print(f"\n=== CONFERMA (3 periodi restanti) ===")
    confirm_rows = []
    for period in CONFIRM_PERIODS:
        b = eval_config(StaircaseParams(mode="baseline"), period, full_data)
        confirm_rows.append(b)
        print(f"  [{period}] baseline: pnl/dd={b['risk_adj_pnl']:.2f} pnl={b['pnl_total']:.0f}")
        for params in survivors:
            row = eval_config(params, period, full_data)
            confirm_rows.append(row)
            print(f"  [{period}] {row['config']:32s} pnl/dd={row['risk_adj_pnl']:.2f} pnl={row['pnl_total']:8.0f}")

    all_rows += confirm_rows
    pd.DataFrame(all_rows).to_csv("results/staircase_full.csv", index=False)
    print("\nCompletato. File in results/.")


if __name__ == "__main__":
    main()
