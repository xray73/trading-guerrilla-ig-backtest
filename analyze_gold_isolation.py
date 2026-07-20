"""
analyze_gold_isolation.py — Isola l'effetto di GOLD sul V6: il test dei
5 periodi (19/07) mostra +4.013,78EUR aggregato (4/5 periodi migliorano)
ma con più del raddoppio dei trade totali — la domanda aperta è quanto
di quel vantaggio viene da GOLD che riempie slot altrimenti VUOTI
(nessun costo, puro guadagno di copertura) e quanto viene da GOLD che
SCALZA un trade DAX/FTSE100 che sarebbe stato migliore (il problema
visto concretamente nel test mirato giugno-luglio 2026, dove GOLD ha
tolto capacità al DAX, che quel mese era il segnale sano).

METODO: replica la selezione multi-candidato di engine_three_asset_gold.py,
ma ogni volta che GOLD viene scelto AL POSTO di un candidato DAX/FTSE100
che sarebbe stato disponibile (competizione reale, non slot vuoto),
simula un "trade ombra" per il candidato escluso — stesso ingresso,
stesso stop/target ATR-based, scansione in avanti sui dati reali fino a
stop/target/scadenza 24h — SENZA aprirlo davvero nel portafoglio (non
tocca il capitale, è un controfattuale).

Scompone il delta di PnL totale (V6+GOLD - V6 baseline) in:
  A) contributo da slot genuinamente vuoti riempiti da GOLD (candidati
     <= slot liberi, nessuna esclusione) — puro guadagno di copertura
  B) contributo netto da competizione reale (candidati > slot liberi):
     PnL controfattuale dei trade DAX/FTSE100 esclusi in quei cicli
     — se grande e positivo, la selezione ha scalzato qualcosa di buono

Nessuna scrittura su D1. Nessuna modifica a engine.py, engine_three_asset_gold.py
o alle altre sottoclassi esistenti — sottoclasse locale isolata.
"""

from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd

import dukascopy_python
from dukascopy_python.instruments import (
    INSTRUMENT_IDX_EUROPE_E_DAAX, INSTRUMENT_IDX_EUROPE_E_FUTSEE_100,
    INSTRUMENT_FX_METALS_XAU_USD,
)

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_three_asset_gold import (
    BacktestEngineV6Gold, instruments_with_gold, _best_subset,
)

CAPITAL_V6 = 1400.0
SYMBOLS_3 = {
    "DAX": INSTRUMENT_IDX_EUROPE_E_DAAX,
    "FTSE100": INSTRUMENT_IDX_EUROPE_E_FUTSEE_100,
    "GOLD": INSTRUMENT_FX_METALS_XAU_USD,
}

PERIODS = [
    ("2015-2016", "2015-01-05", "2016-12-29"),
    ("2020-covid", "2020-01-02", "2020-12-30"),
    ("2023", "2023-01-02", "2023-12-30"),
    ("2024-2025", "2024-01-03", "2025-12-31"),
    ("2026-ytd", "2026-01-05", "2026-07-10"),
]

FULL_FETCH_START = datetime(2014, 10, 1, tzinfo=timezone.utc)
FULL_FETCH_END = datetime(2026, 7, 11, tzinfo=timezone.utc)


def fetch_bars_full(symbol_const) -> pd.DataFrame:
    df = dukascopy_python.fetch(
        symbol_const, dukascopy_python.INTERVAL_MIN_30, dukascopy_python.OFFER_SIDE_BID,
        FULL_FETCH_START, FULL_FETCH_END,
    ).reset_index()
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def slice_period(df: pd.DataFrame, p_start: pd.Timestamp) -> pd.DataFrame:
    return df[df["timestamp"] >= p_start].reset_index(drop=True)


def simulate_shadow_trade(inst_df: pd.DataFrame, entry_idx: int, direction: str,
                           atr_at_entry: float, inst_cfg) -> float:
    """Trade controfattuale: stesso ingresso/stop/target del motore
    reale, scansione in avanti su inst_df fino a stop/target/scadenza
    24h (48 barre). Ritorna il PnL per size=1.0 (unita' di riferimento,
    poi riscalata dal chiamante) — il trade non e' mai aperto davvero."""
    row = inst_df.iloc[entry_idx]
    spread = inst_cfg.spread_fixed
    entry_price = row["open"] + spread / 2 if direction == "long" else row["open"] - spread / 2
    stop_distance = atr_at_entry * inst_cfg.atr_multiplier

    if direction == "long":
        stop = entry_price - stop_distance
        target = entry_price + stop_distance * eng.PARAMS.rr_target
    else:
        stop = entry_price + stop_distance
        target = entry_price - stop_distance * eng.PARAMS.rr_target

    end_idx = min(entry_idx + eng.PARAMS.max_holding_bars, len(inst_df) - 1)
    exit_price = None
    for j in range(entry_idx + 1, end_idx + 1):
        bar = inst_df.iloc[j]
        if direction == "long":
            if bar["low"] <= stop:
                exit_price = stop - spread / 2
                break
            if bar["high"] >= target:
                exit_price = target - spread / 2
                break
        else:
            if bar["high"] >= stop:
                exit_price = stop + spread / 2
                break
            if bar["low"] <= target:
                exit_price = target + spread / 2
                break
    if exit_price is None:
        last_bar = inst_df.iloc[end_idx]
        exit_price = last_bar["close"] - spread / 2 if direction == "long" else last_bar["close"] + spread / 2

    pnl_per_unit = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
    return pnl_per_unit * inst_cfg.point_value


class InstrumentedV6Gold(BacktestEngineV6Gold):
    """Identica a BacktestEngineV6Gold, ma registra per ogni ciclo se
    la selezione ha dovuto escludere candidati (competizione reale) o
    no (slot vuoto riempito), e per i candidati DAX/FTSE100 esclusi a
    favore di GOLD calcola il PnL controfattuale del trade ombra."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.competitive_shadow_pnl = 0.0
        self.n_idle_fill = 0
        self.n_competitive = 0

    def run(self, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        tradable_instruments = [
            name for name in data
            if self.instruments.get(name) is not None and self.instruments[name].tradable
        ]
        all_timestamps = sorted(set().union(
            *[set(data[i]["timestamp"]) for i in tradable_instruments]))

        idx_lookup = {name: data[name].reset_index(drop=True) for name in tradable_instruments}

        for ts in all_timestamps:
            self._reset_day_if_needed(ts)

            for pos in list(self.open_positions):
                inst_df = data[pos.instrument]
                row = inst_df.loc[inst_df["timestamp"] == ts]
                if row.empty:
                    continue
                bar = row.iloc[0]
                bar_index = row.index[0]
                self._try_close_position(pos, bar, bar_index, self.instruments[pos.instrument])

            self.equity_curve.append((ts, self.capital))

            if not self._kill_switch_active and self.open_positions:
                current_bars = {}
                for pos in self.open_positions:
                    inst_df = data[pos.instrument]
                    row = inst_df.loc[inst_df["timestamp"] == ts]
                    if not row.empty:
                        current_bars[pos.instrument] = row.iloc[0]
                perdita_pct = self._floating_loss_pct(current_bars)
                if perdita_pct >= self.p.kill_switch_pct:
                    self._kill_switch_active = True

            if self._kill_switch_active:
                continue
            if self._orders_today >= self.p.max_new_orders_per_day:
                continue
            if len(self.open_positions) >= self.p.max_concurrent_positions:
                continue

            candidates = []
            for name in tradable_instruments:
                inst_df = data[name]
                idx = inst_df.index[inst_df["timestamp"] == ts]
                if len(idx) == 0:
                    continue
                i = idx[0]
                if i == 0:
                    continue
                prev_bar = inst_df.iloc[i - 1]
                cur_bar = inst_df.iloc[i]
                if prev_bar["signal"] not in ("long", "short"):
                    continue
                already_open = any(p.instrument == name for p in self.open_positions)
                if already_open:
                    continue
                candidates.append({
                    "instrument": name, "direction": prev_bar["signal"],
                    "bar": cur_bar, "atr": prev_bar["atr"], "adx": prev_bar["adx"],
                    "rr": self.p.rr_target, "entry_idx": i,
                })

            if not candidates:
                continue

            slots_free = self.p.max_concurrent_positions - len(self.open_positions)
            already_open_instruments = [p.instrument for p in self.open_positions]

            competitive_cycle = len(candidates) > slots_free
            selected = _best_subset(candidates, already_open_instruments, slots_free)
            selected_instruments = {c["instrument"] for c in selected}

            if competitive_cycle:
                self.n_competitive += 1
                excluded = [c for c in candidates if c["instrument"] not in selected_instruments]
                gold_selected = any(c["instrument"] == "GOLD" for c in selected)
                if gold_selected and excluded:
                    for exc in excluded:
                        if exc["instrument"] == "GOLD":
                            continue
                        inst_cfg = self.instruments[exc["instrument"]]
                        shadow_pnl_per_unit = simulate_shadow_trade(
                            idx_lookup[exc["instrument"]], exc["entry_idx"], exc["direction"],
                            exc["atr"], inst_cfg)
                        risk_amount = self.capital * inst_cfg.risk_pct
                        stop_distance = exc["atr"] * inst_cfg.atr_multiplier
                        size_ref = risk_amount / (stop_distance * inst_cfg.point_value) if stop_distance > 0 else 0
                        self.competitive_shadow_pnl += shadow_pnl_per_unit * size_ref
            else:
                self.n_idle_fill += 1

            for c in selected:
                if pd.isna(c["atr"]) or pd.isna(c["adx"]):
                    continue
                if self._orders_today >= self.p.max_new_orders_per_day:
                    break
                self._open_position(c["instrument"], c["direction"], c["bar"], c["atr"], c["adx"])

        trades_df = self.trades_to_dataframe()
        metrics_df = eng.compute_run_metrics(trades_df, self.capital0, self.capital)
        return trades_df, metrics_df


def main():
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=== Isolamento effetto GOLD su V6 — slot vuoti vs competizione reale ===\n")

    instruments_2 = dict(eng.INSTRUMENTS)
    instruments_3 = instruments_with_gold()

    log("Scarico storico DAX/FTSE100/GOLD (fetch unico per strumento)...")
    raw_full = {name: fetch_bars_full(const) for name, const in SYMBOLS_3.items()}
    log("Fatto.\n")

    v6_signals_full = {name: eng.generate_signals(raw_full[name], instruments_3[name]) for name in SYMBOLS_3}

    rows = []
    for label, p_start_str, p_end_str in PERIODS:
        p_start = pd.Timestamp(p_start_str, tz="UTC")
        log(f"Periodo {label}")

        v6_sig_3 = {name: slice_period(v6_signals_full[name], p_start) for name in SYMBOLS_3}
        v6_sig_2_only = {k: v for k, v in v6_sig_3.items() if k != "GOLD"}

        eng_base = BacktestEngineFloatingKillSwitch(capital0=CAPITAL_V6, instruments=instruments_2)
        trades_base, _ = eng_base.run(v6_sig_2_only)
        pnl_base = trades_base["pnl"].sum() if not trades_base.empty else 0.0

        eng_gold = InstrumentedV6Gold(capital0=CAPITAL_V6, instruments=instruments_3)
        trades_gold, _ = eng_gold.run(v6_sig_3)
        pnl_gold_total = trades_gold["pnl"].sum() if not trades_gold.empty else 0.0

        log(f"  V6 baseline PnL: {pnl_base:+.2f}   V6+GOLD PnL: {pnl_gold_total:+.2f}   "
            f"delta: {pnl_gold_total - pnl_base:+.2f}")
        log(f"  Cicli con competizione reale (candidati>slot liberi): {eng_gold.n_competitive}")
        log(f"  Cicli con slot genuinamente vuoto riempito: {eng_gold.n_idle_fill}")
        log(f"  PnL controfattuale (ombra) dei trade DAX/FTSE100 esclusi da GOLD in competizione: "
            f"{eng_gold.competitive_shadow_pnl:+.2f}")
        log(f"  --> Se questo numero e' POSITIVO e grande, GOLD ha scalzato trade che sarebbero "
            f"stati buoni (costo reale). Se e' vicino a zero o negativo, i trade scalzati erano "
            f"comunque deboli (nessun costo reale, o addirittura beneficio).\n")

        rows.append({
            "periodo": label, "pnl_base": pnl_base, "pnl_gold": pnl_gold_total,
            "delta": pnl_gold_total - pnl_base,
            "n_competitive_cycles": eng_gold.n_competitive, "n_idle_cycles": eng_gold.n_idle_fill,
            "shadow_pnl_excluded": eng_gold.competitive_shadow_pnl,
        })

    summary_df = pd.DataFrame(rows)
    import os
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv("results/analyze_gold_isolation.csv", index=False)

    log(f"{'='*70}\nRIEPILOGO — somma sui 5 periodi\n{'='*70}")
    log(f"Delta PnL totale (V6+GOLD - baseline): {summary_df['delta'].sum():+.2f}")
    log(f"Cicli competitivi totali: {summary_df['n_competitive_cycles'].sum()}   "
        f"Cicli slot-vuoto totali: {summary_df['n_idle_cycles'].sum()}")
    log(f"PnL ombra totale dei trade esclusi da GOLD (competizione reale): "
        f"{summary_df['shadow_pnl_excluded'].sum():+.2f}")
    log("\nInterpretazione: se il PnL ombra escluso e' grande e positivo rispetto al delta totale,")
    log("gran parte del vantaggio di GOLD viene comunque eroso da trade DAX/FTSE100 scalzati che")
    log("erano validi — coerente con quanto visto nel test mirato giugno-luglio 2026.")

    with open("results/analyze_gold_isolation.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. ===")


if __name__ == "__main__":
    main()
