"""
exit_variants_test.py — Testa due modifiche alla gestione dello stop
DOPO l'apertura del trade (lato uscita, non ingresso — quindi il numero
di trade per costruzione resta identico al baseline, isola l'effetto
sull'uscita):

  1. BREAKEVEN: quando il trade è a favore di r_threshold × R, lo stop si
     sposta al prezzo di ingresso + offset_r × R (mai oltre il pareggio
     puro, per garantire una vittoria vera che copra lo spread).
  2. TRAILING: come sopra, ma dopo lo spostamento a breakeven lo stop
     continua a seguire il prezzo a distanza trail_atr_mult × ATR
     dall'estremo favorevole raggiunto.

Il target originale (2R, Charter sez.3) resta INVARIATO come tetto in
entrambi i casi: se il prezzo va dritto al target senza mai ritracciare,
il trade chiude lì esattamente come nel motore v2 di produzione. Il
meccanismo nuovo interviene SOLO se il prezzo ritraccia prima di
arrivarci.

Matrice testata (13 configurazioni):
  - baseline (motore v2 invariato)
  - breakeven: r_threshold in {0.5, 1.0, 1.5} — 3 varianti
  - trailing: r_threshold in {0.5, 1.0, 1.5} x trail_atr_mult in
    {1.0, 1.5, 2.0} — 9 varianti

Criterio di promozione (fissato in sessione, PRIMA di vedere risultati):
  - win_rate >= baseline_win_rate + 2 punti percentuali
  - profit_factor >= 90% del profit_factor baseline (evita "win rate su
    ma qualità delle vincite troppo compromessa")
Nota importante scoperta in fase di test (non assunta a priori): il numero
di trade NON è garantito identico al baseline, anche toccando solo
l'uscita — se un trade chiude prima o dopo per via del nuovo stop, questo
libera o occupa uno slot di posizione prima/dopo (il motore ammette max
1-2 posizioni concorrenti, mai 2 sullo stesso strumento), il che può
abilitare o bloccare un segnale successivo che altrimenti sarebbe andato
diversamente. L'effetto è tipicamente modesto (qualche punto percentuale)
ma va riportato, non assunto pari a zero.

Disciplina walk-forward identica al resto della sessione: selezione su
2023 (train) → verifica 2024-2025 (test, stessi criteri, nessun ritocco)
→ conferma sui 3 periodi restanti (solo riportata).

Limite dichiarato: come il resto del motore, i controlli di stop/target
sulla stessa barra usano high e low della barra corrente senza sapere
il reale ordine cronologico intra-barra (stesso limite già presente nel
motore di produzione per stop vs target sulla stessa barra — non è una
nuova forma di approssimazione, è la stessa già accettata altrove).
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

R_THRESHOLDS = [0.5, 1.0, 1.5]
TRAIL_ATR_MULTS = [1.0, 1.5, 2.0]
OFFSET_R = 0.1  # fisso, non in griglia — copre lo spread con ampio margine

WIN_RATE_MARGIN_PP = 0.02   # fissato in sessione
PF_FLOOR_PCT = 0.90         # fissato in sessione

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
# 1. FETCH DATI (REST D1, stesso pattern degli altri script della sessione)
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
# 2. MOTORE ESTESO — stop dinamico (breakeven / trailing), tutto il
#    resto identico a eng.BacktestEngine (nessuna modifica al motore
#    di produzione)
# =====================================================================

@dataclasses.dataclass
class ExitParams:
    mode: str  # "baseline" | "breakeven" | "trailing"
    r_threshold: float = 1.0
    trail_atr_mult: float = 1.5
    offset_r: float = OFFSET_R

    def label(self) -> str:
        if self.mode == "baseline":
            return "baseline"
        if self.mode == "breakeven":
            return f"breakeven_r{self.r_threshold}"
        return f"trailing_r{self.r_threshold}_atr{self.trail_atr_mult}"


class ExitVariantEngine(eng.BacktestEngine):
    def __init__(self, capital0: float, exit_params: ExitParams,
                 p: eng.ChartaParams = eng.PARAMS,
                 instruments: dict = eng.INSTRUMENTS):
        super().__init__(capital0=capital0, p=p, instruments=instruments)
        self.exit_params = exit_params
        self._pos_state: dict[int, dict] = {}

    def _open_position(self, instrument, direction, bar, atr_at_entry, adx_at_entry):
        n_before = len(self.open_positions)
        super()._open_position(instrument, direction, bar, atr_at_entry, adx_at_entry)
        if len(self.open_positions) > n_before:
            pos = self.open_positions[-1]
            r_unit = abs(pos.entry_price - pos.stop_loss)
            self._pos_state[id(pos)] = {
                "extreme_price": pos.entry_price,
                "breakeven_moved": False,
                "r_unit": r_unit,
            }

    def _try_close_position(self, pos, bar, bar_index, inst):
        if self.exit_params.mode != "baseline":
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

        ep = self.exit_params

        if not state["breakeven_moved"] and favorable >= ep.r_threshold * r_unit:
            offset = ep.offset_r * r_unit
            if pos.direction == "long":
                pos.stop_loss = max(pos.stop_loss, pos.entry_price + offset)
            else:
                pos.stop_loss = min(pos.stop_loss, pos.entry_price - offset)
            state["breakeven_moved"] = True

        if ep.mode == "trailing" and state["breakeven_moved"]:
            trail_dist = ep.trail_atr_mult * pos.atr_at_entry
            if pos.direction == "long":
                pos.stop_loss = max(pos.stop_loss, state["extreme_price"] - trail_dist)
            else:
                pos.stop_loss = min(pos.stop_loss, state["extreme_price"] + trail_dist)


# =====================================================================
# 3. VALUTAZIONE DI UNA CONFIGURAZIONE SU UN PERIODO
# =====================================================================

def eval_config(exit_params: ExitParams, period_label: str,
                 full_data: dict[str, pd.DataFrame]) -> dict:
    data = {}
    for name, full_df in full_data.items():
        inst = eng.INSTRUMENTS[name]
        window, period_start = slice_period(full_df, period_label)
        sig = eng.generate_signals(window, inst)  # segnale INVARIATO, tocchiamo solo l'uscita
        sig = trim_warmup(sig, period_start)
        data[name] = sig

    engine_ = ExitVariantEngine(capital0=CAPITAL0, exit_params=exit_params)
    trades_df, metrics_df = engine_.run(data)

    num_trades = int(metrics_df["num_trades"].iloc[0])
    return {
        "config": exit_params.label(),
        "mode": exit_params.mode,
        "r_threshold": exit_params.r_threshold if exit_params.mode != "baseline" else None,
        "trail_atr_mult": exit_params.trail_atr_mult if exit_params.mode == "trailing" else None,
        "period": period_label,
        "num_trades": num_trades,
        "win_rate": float(metrics_df["win_rate"].iloc[0]) if num_trades else 0.0,
        "pnl_total": float(metrics_df["pnl_total"].iloc[0]),
        "profit_factor": float(metrics_df["profit_factor"].iloc[0]) if num_trades else np.nan,
        "expectancy": float(metrics_df["expectancy"].iloc[0]) if num_trades else np.nan,
        "max_drawdown_pct": float(metrics_df["max_drawdown_pct"].iloc[0]) if num_trades else np.nan,
    }


def build_grid() -> list[ExitParams]:
    grid = [ExitParams(mode="baseline")]
    for rt in R_THRESHOLDS:
        grid.append(ExitParams(mode="breakeven", r_threshold=rt))
    for rt in R_THRESHOLDS:
        for tm in TRAIL_ATR_MULTS:
            grid.append(ExitParams(mode="trailing", r_threshold=rt, trail_atr_mult=tm))
    return grid


def passes_criteria(row: dict, baseline_row: dict) -> bool:
    if row["num_trades"] == 0:
        return False
    wr_ok = row["win_rate"] >= baseline_row["win_rate"] + WIN_RATE_MARGIN_PP
    pf_ok = (row["profit_factor"] >= baseline_row["profit_factor"] * PF_FLOOR_PCT
             if not np.isnan(row["profit_factor"]) and not np.isnan(baseline_row["profit_factor"])
             else False)
    return wr_ok and pf_ok


# =====================================================================
# 4. MAIN — TRAIN -> SELEZIONE -> TEST -> CONFERMA
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
    print(f"\n{len(grid)} configurazioni totali (1 baseline + {len(R_THRESHOLDS)} breakeven + "
          f"{len(R_THRESHOLDS)*len(TRAIL_ATR_MULTS)} trailing)")

    all_rows = []

    # --- TRAIN ---
    print(f"\n=== TRAIN ({TRAIN_PERIOD}) ===")
    train_rows = []
    for ep in grid:
        row = eval_config(ep, TRAIN_PERIOD, full_data)
        train_rows.append(row)
        print(f"  {row['config']:24s} trades={row['num_trades']:4d} "
              f"win_rate={row['win_rate']*100:5.2f}% pf={row['profit_factor']:.3f} "
              f"pnl={row['pnl_total']:8.0f}")
    all_rows += train_rows

    baseline_train = next(r for r in train_rows if r["mode"] == "baseline")
    print(f"\n  Baseline train: win_rate={baseline_train['win_rate']*100:.2f}% "
          f"pf={baseline_train['profit_factor']:.3f}")
    print(f"  Soglia promozione: win_rate >= {(baseline_train['win_rate']+WIN_RATE_MARGIN_PP)*100:.2f}% "
          f"E profit_factor >= {baseline_train['profit_factor']*PF_FLOOR_PCT:.3f}")

    candidates = [r for r in train_rows if r["mode"] != "baseline" and passes_criteria(r, baseline_train)]

    pd.DataFrame(all_rows).to_csv("results/exit_variants_train.csv", index=False)

    if not candidates:
        print("\n  NESSUNA configurazione promossa al test. Chiuso qui, come da criterio fissato.")
        return

    print(f"\n  PROMOSSE AL TEST ({len(candidates)}): " + ", ".join(c["config"] for c in candidates))

    # --- TEST ---
    print(f"\n=== TEST ({TEST_PERIOD}, out-of-sample, nessun ritocco) ===")
    test_rows = []
    baseline_test = eval_config(ExitParams(mode="baseline"), TEST_PERIOD, full_data)
    test_rows.append(baseline_test)
    print(f"  baseline: win_rate={baseline_test['win_rate']*100:.2f}% pf={baseline_test['profit_factor']:.3f}")

    survivors = []
    grid_by_label = {ep.label(): ep for ep in grid}
    for c in candidates:
        ep = grid_by_label[c["config"]]
        row = eval_config(ep, TEST_PERIOD, full_data)
        test_rows.append(row)
        passed = passes_criteria(row, baseline_test)
        print(f"  {row['config']:24s} win_rate={row['win_rate']*100:5.2f}% "
              f"pf={row['profit_factor']:.3f}  {'SUPERA' if passed else 'non supera'}")
        if passed:
            survivors.append(ep)

    all_rows += test_rows
    pd.DataFrame(all_rows).to_csv("results/exit_variants_train_test.csv", index=False)

    if not survivors:
        print("\n  NESSUNA configurazione supera il test out-of-sample. "
              "Pattern train-vince/test-crolla — stesso esito di altri tentativi già scartati. Chiuso qui.")
        return

    print(f"\n  SOPRAVVISSUTE AL TEST ({len(survivors)}): " + ", ".join(s.label() for s in survivors))

    # --- CONFIRM ---
    print(f"\n=== CONFERMA (3 periodi restanti, solo riportata) ===")
    confirm_rows = []
    for period in CONFIRM_PERIODS:
        b = eval_config(ExitParams(mode="baseline"), period, full_data)
        confirm_rows.append(b)
        print(f"  [{period}] baseline: win_rate={b['win_rate']*100:.2f}% pf={b['profit_factor']:.3f}")
        for ep in survivors:
            row = eval_config(ep, period, full_data)
            confirm_rows.append(row)
            print(f"  [{period}] {row['config']:24s} win_rate={row['win_rate']*100:5.2f}% "
                  f"pf={row['profit_factor']:.3f} pnl={row['pnl_total']:8.0f}")

    all_rows += confirm_rows
    pd.DataFrame(all_rows).to_csv("results/exit_variants_full.csv", index=False)

    print("\nCompletato. File in results/.")


if __name__ == "__main__":
    main()
