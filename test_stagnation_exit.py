"""
test_stagnation_exit.py — TEST 4/6 del programma di validazione (21/07/2026).

Domanda: chiudere anticipatamente un trade che a una barra di controllo
fissa non ha ancora superato una soglia minima di R (segno che "non e'
mai partito", pattern robusto trovato su tutta la popolazione — non un
sottogruppo piccolo come il give-back del Test 3) riduce le perdite
piu' di quanto tagli i guadagni?

DIVERSO dal give-back (Test 3): qui il controllo NON dipende da un
picco gia' raggiunto — e' un controllo fisso a bar_offset==check_bar,
"sei sopra o sotto soglia adesso?". Verificato in chat prima di
costruire il motore: guardando SIA vincitori che perdenti sotto soglia
(0.2R) su una griglia di barre, i perdenti restano stabili al 60-68% in
tutte le barre, mentre i "falsi positivi" tra i vincitori CALANO da
33% (barra 1) a ~20% (barra 10+) — aspettare di piu' prima del
controllo cattura una quota simile di perdenti con meno vincitori
sacrificati.

DUE varianti testate in PARALLELO, nessuna scelta dopo aver visto i
risultati (entrambe portate avanti fino in fondo, bootstrap completo
per entrambe, poi confronto):
  A) check_bar=3,  threshold_r=0.2  (aggressiva, taglio rapido)
  B) check_bar=10, threshold_r=0.2  (paziente, meno falsi positivi attesi)

Motore: nuova sottoclasse isolata di BacktestEngineFloatingKillSwitch.
SANITY CHECK (una volta sola, verifica di correttezza del motore, non
selezione tra le varianti): check_bar=None deve produrre risultati
IDENTICI al motore standard.

Nessuna scrittura su trades/backtest_runs/live_*/research_v6_*. Output
SOLO aggregati.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch

CAPITAL0 = 2000.0
SYMBOLS = ["DAX", "FTSE100"]
VARIANTS = [
    ("A_bar3", 3, 0.2),
    ("B_bar10", 10, 0.2),
]
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
# 1) MOTORE — sottoclasse isolata, uscita anticipata per stagnazione
# =====================================================================

class BacktestEngineStagnationExit(BacktestEngineFloatingKillSwitch):
    def __init__(self, *args, check_bar: int = None, threshold_r: float = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_bar = check_bar
        self.threshold_r = threshold_r

    def _check_stagnation(self, pos, bar, bar_index, inst) -> bool:
        if self.check_bar is None or self.threshold_r is None:
            return False
        bars_held = bar_index - pos.entry_bar_index
        if bars_held != self.check_bar:
            return False
        stop_distance = pos.atr_at_entry * inst.atr_multiplier
        if stop_distance <= 0:
            return False
        if pos.direction == "long":
            current_r = (bar["close"] - pos.entry_price) / stop_distance
        else:
            current_r = (pos.entry_price - bar["close"]) / stop_distance

        if current_r < self.threshold_r:
            spread = inst.spread_fixed
            exit_price = (bar["close"] - spread / 2 if pos.direction == "long"
                          else bar["close"] + spread / 2)
            self._close_position(pos, bar["timestamp"], exit_price, "early_stagnation")
            return True
        return False

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
                inst = self.instruments[pos.instrument]

                closed = self._try_close_position(pos, bar, bar_index, inst)
                if closed:
                    continue

                # --- NUOVO: controllo stagnazione a barra fissa ---
                self._check_stagnation(pos, bar, bar_index, inst)
                # --- fine nuovo ---

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
# 2) BOOTSTRAP a blocchi di giornata (stesso protocollo di GOLD/Test 1/2/3)
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


def replay_path(day_sequence: list, day_index: dict, capital0: float) -> float:
    capital = capital0
    for day in day_sequence:
        trades_today = day_index.get(day, [])
        if not trades_today:
            continue
        day_start_capital = capital
        kill_switch_active = False
        for risk_pct, r_mult in trades_today:
            if kill_switch_active:
                continue
            risk_amount = capital * risk_pct
            pnl = r_mult * risk_amount
            capital += pnl
            daily_pnl_pct = (capital - day_start_capital) / day_start_capital if day_start_capital else 0.0
            if daily_pnl_pct <= -KILL_SWITCH_PCT:
                kill_switch_active = True
    return capital - capital0


def bootstrap_period(base_trades: pd.DataFrame, variant_trades: pd.DataFrame,
                      all_days: list, capital0: float, n_boot: int,
                      rng: np.random.Generator) -> dict:
    base_idx = build_day_index(base_trades)
    variant_idx = build_day_index(variant_trades)
    n_days = len(all_days)

    observed_base = replay_path(all_days, base_idx, capital0)
    observed_variant = replay_path(all_days, variant_idx, capital0)
    observed_delta = observed_variant - observed_base

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        sampled_days = [all_days[i] for i in rng.integers(0, n_days, size=n_days)]
        pnl_base = replay_path(sampled_days, base_idx, capital0)
        pnl_variant = replay_path(sampled_days, variant_idx, capital0)
        deltas[b] = pnl_variant - pnl_base

    return {
        "observed_delta": observed_delta, "boot_mean": deltas.mean(), "boot_sd": deltas.std(),
        "ci_low_95": np.percentile(deltas, 2.5), "ci_high_95": np.percentile(deltas, 97.5),
        "pct_iter_non_positive": (deltas <= 0).mean() * 100, "deltas": deltas,
    }


# =====================================================================
# 3) MAIN — sanity check (una volta) -> entrambe le varianti fino in fondo -> confronto
# =====================================================================

def slice_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)


def run_variant(label, check_bar, threshold_r, signals_full, rng):
    """Rigioca la variante sui 5 periodi ufficiali e ritorna il riepilogo
    aggregato + il dettaglio per periodo."""
    rows = []
    total_deltas = np.zeros(N_BOOT)
    total_observed_delta = 0.0
    total_trades_base = 0
    total_trades_variant = 0
    total_early_exits = 0

    for period_label, start_str, end_str in PERIODS:
        start = pd.Timestamp(start_str, tz="UTC")
        end = pd.Timestamp(end_str, tz="UTC") + pd.Timedelta(days=1) if end_str else pd.Timestamp.now(tz="UTC")
        sig_period = {name: slice_period(signals_full[name], start, end) for name in SYMBOLS}

        eng_base = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0, instruments=eng.INSTRUMENTS)
        trades_base, metrics_base = eng_base.run(sig_period)

        eng_variant = BacktestEngineStagnationExit(
            capital0=CAPITAL0, instruments=eng.INSTRUMENTS,
            check_bar=check_bar, threshold_r=threshold_r)
        trades_variant, metrics_variant = eng_variant.run(sig_period)

        n_early = (trades_variant["exit_reason"] == "early_stagnation").sum() if not trades_variant.empty else 0
        total_early_exits += n_early

        mb, mv = metrics_base.iloc[0], metrics_variant.iloc[0]
        print(f"  [{label}] {period_label}: baseline={len(trades_base)} trade pnl={mb['pnl_total']:+.2f}  |  "
              f"variante={len(trades_variant)} trade ({n_early} usciti per stagnazione) pnl={mv['pnl_total']:+.2f}")

        all_days = sorted(set(sig_period["DAX"]["timestamp"].dt.date) |
                           set(sig_period["FTSE100"]["timestamp"].dt.date))
        result = bootstrap_period(trades_base, trades_variant, all_days, CAPITAL0, N_BOOT, rng)
        print(f"      bootstrap: delta={result['observed_delta']:+.2f}  sd={result['boot_sd']:.2f}  "
              f"IC95%=[{result['ci_low_95']:+.2f}, {result['ci_high_95']:+.2f}]  "
              f"%iter<=0={result['pct_iter_non_positive']:.1f}%")

        rows.append({
            "variante": label, "periodo": period_label,
            "n_trade_base": len(trades_base), "n_trade_variant": len(trades_variant),
            "n_early_exits": int(n_early),
            "pnl_base": mb["pnl_total"], "pnl_variant": mv["pnl_total"],
            "delta_osservato": result["observed_delta"], "boot_sd": result["boot_sd"],
            "ci95_low": result["ci_low_95"], "ci95_high": result["ci_high_95"],
            "pct_iter_delta_non_positivo": result["pct_iter_non_positive"],
        })
        total_deltas += result["deltas"]
        total_observed_delta += result["observed_delta"]
        total_trades_base += len(trades_base)
        total_trades_variant += len(trades_variant)

    z_tot = total_observed_delta / total_deltas.std() if total_deltas.std() > 0 else float("nan")
    ci_low_tot, ci_high_tot = np.percentile(total_deltas, [2.5, 97.5])
    pct_non_positive_tot = (total_deltas <= 0).mean() * 100

    summary = {
        "variante": label, "check_bar": check_bar, "threshold_r": threshold_r,
        "trade_totali_base": total_trades_base, "trade_totali_variant": total_trades_variant,
        "uscite_anticipate_totali": total_early_exits,
        "delta_osservato_totale": total_observed_delta,
        "boot_sd_totale": total_deltas.std(), "ci95_low_totale": ci_low_tot, "ci95_high_totale": ci_high_tot,
        "pct_iter_non_positivo_totale": pct_non_positive_tot, "z_approx": z_tot,
    }
    return rows, summary


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

    # --- SANITY CHECK (una volta sola, verifica di correttezza del motore) ---
    print("=== SANITY CHECK (check_bar=None deve dare risultati identici al motore standard) ===")
    p_start, p_end = pd.Timestamp("2023-01-02", tz="UTC"), pd.Timestamp("2023-12-31", tz="UTC")
    sig_check = {name: slice_period(signals_full[name], p_start, p_end) for name in SYMBOLS}

    baseline_check = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0, instruments=eng.INSTRUMENTS)
    trades_bc, _ = baseline_check.run(sig_check)
    variant_check = BacktestEngineStagnationExit(
        capital0=CAPITAL0, instruments=eng.INSTRUMENTS, check_bar=None, threshold_r=None)
    trades_vc, _ = variant_check.run(sig_check)

    same_n = len(trades_bc) == len(trades_vc)
    same_capital = abs(baseline_check.capital - variant_check.capital) < 0.01
    same_pnl = same_n and np.allclose(trades_bc["pnl"].values, trades_vc["pnl"].values, atol=1e-6)
    if same_n and same_capital and same_pnl:
        print(f"SANITY CHECK: PASS ({len(trades_bc)} trade identici, capitale identico)\n")
    else:
        print(f"SANITY CHECK: FALLITO — baseline={len(trades_bc)} trade/{baseline_check.capital:.2f}, "
              f"variante={len(trades_vc)} trade/{variant_check.capital:.2f}. STOP, non procedo oltre.")
        return

    # --- ENTRAMBE LE VARIANTI, portate avanti fino in fondo, nessuna selezione anticipata ---
    rng = np.random.default_rng(SEED)
    all_rows = []
    all_summaries = []
    for label, check_bar, threshold_r in VARIANTS:
        print(f"=== Variante {label}: check_bar={check_bar}, threshold_r={threshold_r} ===")
        rows, summary = run_variant(label, check_bar, threshold_r, signals_full, rng)
        all_rows.extend(rows)
        all_summaries.append(summary)
        print()

    # --- CONFRONTO FINALE, entrambe le varianti fianco a fianco ---
    print(f"{'='*70}\nCONFRONTO FINALE — entrambe le varianti, aggregato sui 5 periodi\n{'='*70}")
    for s in all_summaries:
        print(f"\n{s['variante']} (check_bar={s['check_bar']}, threshold_r={s['threshold_r']}):")
        print(f"  trade totali: baseline={s['trade_totali_base']}  variante={s['trade_totali_variant']}  "
              f"(di cui {s['uscite_anticipate_totali']} usciti per stagnazione)")
        print(f"  delta osservato totale: {s['delta_osservato_totale']:+.2f} EUR")
        print(f"  IC 95% del rumore atteso: [{s['ci95_low_totale']:+.2f}, {s['ci95_high_totale']:+.2f}]")
        print(f"  % iterazioni bootstrap con delta<=0: {s['pct_iter_non_positivo_totale']:.1f}%")
        print(f"  z-score approssimato: {s['z_approx']:.2f}")

    print(f"\nVERDETTO: se il delta osservato cade dentro l'IC 95% del rumore, o se >10-15% delle "
          f"iterazioni danno delta<=0, la variante NON e' chiaramente distinguibile dal rumore "
          f"campionario (Protocollo Anti-Rumore, principio 4) — stesso standard usato per tutti i test precedenti.")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(all_rows).to_csv("results/test_stagnation_exit_dettaglio.csv", index=False)
    pd.DataFrame(all_summaries).to_csv("results/test_stagnation_exit_riepilogo.csv", index=False)
    print("\nDettaglio salvato in results/test_stagnation_exit_dettaglio.csv "
          "e results/test_stagnation_exit_riepilogo.csv")


if __name__ == "__main__":
    main()
