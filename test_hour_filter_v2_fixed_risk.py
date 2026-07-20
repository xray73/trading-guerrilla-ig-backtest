"""
test_hour_filter_v2_fixed_risk.py — RIPETIZIONE del Test 1 (filtro
orario 19-22 UTC) con bootstrap a RISCHIO FISSO invece che a
compounding.

CORREZIONE DI METODO (21/07/2026): il bootstrap usato in GOLD e nei
Test 1-4 ricalcolava il rischio di ogni trade sul capitale CORRENTE
durante il ricampionamento — questo introduce un rumore aggiuntivo
legato SOLO all'ordine in cui i giorni ricampionati capitano (se i
giorni fortunati capitano prima, il capitale cresce prima, le
scommesse successive sono piu' grandi e il risultato si amplifica),
non al contenuto dei trade stessi. Verificato con un prototipo su dati
sintetici (10 seed diversi, due scale diverse): il bootstrap a rischio
fisso riduce la deviazione standard del rumore di un fattore 2-3x.

Questo script rigioca lo STESSO identico Test 1 (motore, sanity check,
soglie 19-22 UTC, 5 periodi ufficiali) ma con la nuova funzione di
bootstrap a rischio fisso (risk_amount = capital0 * risk_pct, MAI
ricalcolato sul capitale corrente durante il resampling). Se il quadro
cambia, conferma che il problema era nel metro di misura, non nelle
idee testate finora.

Nessuna modifica a engine.py o alle sottoclassi esistenti. Nessuna
scrittura su trades/backtest_runs/live_*/research_v6_*. Output SOLO
aggregati.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch

CAPITAL0 = 2000.0
SYMBOLS = ["DAX", "FTSE100"]
AVOID_HOURS = {19, 20, 21, 22}
N_BOOT = 2000
SEED = 20260721
KILL_SWITCH_PCT = eng.PARAMS.kill_switch_pct

PERIODS = [
    ("2015-2016", "2015-01-05", "2016-12-29"),
    ("2020-covid", "2020-01-02", "2020-12-30"),
    ("2023", "2023-01-02", "2023-12-30"),
    ("2024-2025", "2024-01-03", "2025-12-31"),
    ("2026-ytd", "2026-01-05", None),
]


# =====================================================================
# 1) MOTORE — IDENTICO al Test 1 originale, nessuna modifica
# =====================================================================

class BacktestEngineHourFilter(BacktestEngineFloatingKillSwitch):
    def __init__(self, *args, avoid_hours: set[int] = frozenset(), **kwargs):
        super().__init__(*args, **kwargs)
        self.avoid_hours = avoid_hours

    def run(self, data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        tradable_instruments = [
            name for name in data
            if self.instruments.get(name) is not None and self.instruments[name].tradable
        ]
        if not tradable_instruments:
            raise ValueError("Nessuno strumento tradabile fornito a run().")

        all_timestamps = sorted(set().union(
            *[set(data[i]["timestamp"]) for i in tradable_instruments]))

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
                if int(cur_bar["hour_utc"]) in self.avoid_hours:
                    continue
                already_open = any(p.instrument == name for p in self.open_positions)
                if already_open:
                    continue
                candidates.append({
                    "instrument": name, "direction": prev_bar["signal"],
                    "bar": cur_bar, "atr": prev_bar["atr"], "adx": prev_bar["adx"],
                    "rr": self.p.rr_target,
                })

            if not candidates:
                continue

            candidates.sort(key=lambda c: (-c["rr"], self._correlation_penalty(c["instrument"])))

            slots_free = self.p.max_concurrent_positions - len(self.open_positions)
            for c in candidates:
                if slots_free <= 0:
                    break
                if self._orders_today >= self.p.max_new_orders_per_day:
                    break
                if pd.isna(c["atr"]) or pd.isna(c["adx"]):
                    continue
                self._open_position(c["instrument"], c["direction"], c["bar"],
                                     c["atr"], c["adx"])
                slots_free -= 1

        trades_df = self.trades_to_dataframe()
        metrics_df = eng.compute_run_metrics(trades_df, self.capital0, self.capital)
        return trades_df, metrics_df


# =====================================================================
# 2) BOOTSTRAP A RISCHIO FISSO — la correzione
# =====================================================================

def build_day_index(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {}
    df = trades_df.copy()
    df["entry_day"] = pd.to_datetime(df["entry_time"]).dt.date
    df["risk_pct_used"] = df["instrument"].map(
        {name: inst.risk_pct for name, inst in eng.INSTRUMENTS.items()})
    df = df.sort_values("entry_time")
    out = {}
    for day, grp in df.groupby("entry_day"):
        out[day] = list(zip(grp["risk_pct_used"], grp["r_multiple"]))
    return out


def replay_path_fixed_risk(day_sequence: list, day_index: dict, capital0: float) -> float:
    """CORREZIONE: risk_amount = capital0 * risk_pct, sempre FISSO — mai
    ricalcolato sul capitale corrente. Isola l'effetto del contenuto dei
    trade dall'effetto (spurio) dell'ordine di ricampionamento."""
    total_pnl = 0.0
    for day in day_sequence:
        trades_today = day_index.get(day, [])
        if not trades_today:
            continue
        day_pnl = 0.0
        kill_switch_active = False
        for risk_pct, r_mult in trades_today:
            if kill_switch_active:
                continue
            risk_amount = capital0 * risk_pct
            pnl = r_mult * risk_amount
            day_pnl += pnl
            if (day_pnl / capital0) <= -KILL_SWITCH_PCT:
                kill_switch_active = True
        total_pnl += day_pnl
    return total_pnl


def bootstrap_period(base_trades: pd.DataFrame, variant_trades: pd.DataFrame,
                      all_days: list, capital0: float, n_boot: int,
                      rng: np.random.Generator) -> dict:
    base_idx = build_day_index(base_trades)
    variant_idx = build_day_index(variant_trades)
    n_days = len(all_days)

    observed_base = replay_path_fixed_risk(all_days, base_idx, capital0)
    observed_variant = replay_path_fixed_risk(all_days, variant_idx, capital0)
    observed_delta = observed_variant - observed_base

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        sampled_days = [all_days[i] for i in rng.integers(0, n_days, size=n_days)]
        pnl_base = replay_path_fixed_risk(sampled_days, base_idx, capital0)
        pnl_variant = replay_path_fixed_risk(sampled_days, variant_idx, capital0)
        deltas[b] = pnl_variant - pnl_base

    return {
        "observed_delta": observed_delta, "boot_mean": deltas.mean(), "boot_sd": deltas.std(),
        "ci_low_95": np.percentile(deltas, 2.5), "ci_high_95": np.percentile(deltas, 97.5),
        "pct_iter_non_positive": (deltas <= 0).mean() * 100, "deltas": deltas,
    }


# =====================================================================
# 3) MAIN
# =====================================================================

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

    print("=== SANITY CHECK (avoid_hours=set() deve dare risultati identici al motore standard) ===")
    p_start, p_end = pd.Timestamp("2023-01-02", tz="UTC"), pd.Timestamp("2023-12-31", tz="UTC")
    sig_check = {name: slice_period(signals_full[name], p_start, p_end) for name in SYMBOLS}
    baseline_check = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0, instruments=eng.INSTRUMENTS)
    trades_bc, _ = baseline_check.run(sig_check)
    variant_check = BacktestEngineHourFilter(capital0=CAPITAL0, instruments=eng.INSTRUMENTS, avoid_hours=set())
    trades_vc, _ = variant_check.run(sig_check)
    same_n = len(trades_bc) == len(trades_vc)
    same_capital = abs(baseline_check.capital - variant_check.capital) < 0.01
    same_pnl = same_n and np.allclose(trades_bc["pnl"].values, trades_vc["pnl"].values, atol=1e-6)
    if same_n and same_capital and same_pnl:
        print(f"SANITY CHECK: PASS ({len(trades_bc)} trade identici)\n")
    else:
        print("SANITY CHECK: FALLITO. STOP.")
        return

    print("=== RUN sui 5 periodi ufficiali (bootstrap A RISCHIO FISSO) ===")
    rng = np.random.default_rng(SEED)
    rows = []
    total_deltas = np.zeros(N_BOOT)
    total_observed_delta = 0.0

    for label, start_str, end_str in PERIODS:
        start = pd.Timestamp(start_str, tz="UTC")
        end = pd.Timestamp(end_str, tz="UTC") + pd.Timedelta(days=1) if end_str else pd.Timestamp.now(tz="UTC")
        print(f"--- Periodo {label} ---")
        sig_period = {name: slice_period(signals_full[name], start, end) for name in SYMBOLS}

        eng_base = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0, instruments=eng.INSTRUMENTS)
        trades_base, metrics_base = eng_base.run(sig_period)
        eng_variant = BacktestEngineHourFilter(capital0=CAPITAL0, instruments=eng.INSTRUMENTS,
                                                avoid_hours=AVOID_HOURS)
        trades_variant, metrics_variant = eng_variant.run(sig_period)

        mb, mv = metrics_base.iloc[0], metrics_variant.iloc[0]
        print(f"  baseline: {len(trades_base)} trade, pnl={mb['pnl_total']:+.2f}")
        print(f"  variante: {len(trades_variant)} trade, pnl={mv['pnl_total']:+.2f}")

        all_days = sorted(set(sig_period["DAX"]["timestamp"].dt.date) |
                           set(sig_period["FTSE100"]["timestamp"].dt.date))
        result = bootstrap_period(trades_base, trades_variant, all_days, CAPITAL0, N_BOOT, rng)
        print(f"  bootstrap (rischio fisso): delta={result['observed_delta']:+.2f}  "
              f"sd={result['boot_sd']:.2f}  IC95%=[{result['ci_low_95']:+.2f}, {result['ci_high_95']:+.2f}]  "
              f"%iter<=0={result['pct_iter_non_positive']:.1f}%\n")

        rows.append({
            "periodo": label, "n_trade_base": len(trades_base), "n_trade_variant": len(trades_variant),
            "delta_osservato": result["observed_delta"], "boot_sd": result["boot_sd"],
            "ci95_low": result["ci_low_95"], "ci95_high": result["ci_high_95"],
            "pct_iter_delta_non_positivo": result["pct_iter_non_positive"],
        })
        total_deltas += result["deltas"]
        total_observed_delta += result["observed_delta"]

    ci_low_tot, ci_high_tot = np.percentile(total_deltas, [2.5, 97.5])
    pct_non_positive_tot = (total_deltas <= 0).mean() * 100
    z_tot = total_observed_delta / total_deltas.std() if total_deltas.std() > 0 else float("nan")

    print(f"{'='*70}\nAGGREGATO (bootstrap a rischio fisso)\n{'='*70}")
    print(f"Delta osservato totale: {total_observed_delta:+.2f} EUR")
    print(f"Bootstrap: sd={total_deltas.std():.2f}  IC95%=[{ci_low_tot:+.2f}, {ci_high_tot:+.2f}]")
    print(f"% iterazioni con delta<=0: {pct_non_positive_tot:.1f}%")
    print(f"z-score approssimato: {z_tot:.2f}")
    print(f"\nCONFRONTO CON IL TEST 1 ORIGINALE (bootstrap a compounding, per riferimento):")
    print(f"  delta osservato era lo stesso tipo di numero, ma sd_totale era ~7430 EUR, z~-0.30")
    print(f"  qui sd_totale={total_deltas.std():.2f} EUR — se molto piu' piccola, conferma il problema di metodo")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/test_hour_filter_v2_fixed_risk.csv", index=False)
    print("\nDettaglio salvato in results/test_hour_filter_v2_fixed_risk.csv")


if __name__ == "__main__":
    main()
