"""
test_no_overnight.py — TEST 5/6 del programma di validazione (21/07/2026).

Domanda: chiudere forzatamente ogni posizione al primo cutoff overnight
(22:00 UK time circa, invece di lasciarla aperta fino a stop/target/
max_holding naturale) migliora il risultato netto, evitando il costo
del funding overnight IG?

Origine: conto grezzo su dati gia' estratti (21/07/2026) — sui 512
trade su 2.156 che attraversano almeno un cutoff, chiudere forzatamente
avrebbe reso +1.134€ meglio del comportamento attuale con funding pagato
(costo funding 2.370€ > valore lasciato sul tavolo 1.236€ chiudendo
prima). Questo test verifica la stessa domanda con un motore causale
vero (che puo' anche liberare uno slot prima e aprire un trade diverso
al posto di quello chiuso — effetto che il conto grezzo statico non
poteva catturare).

REGOLA ISOLATA — nessuna modifica a target/stop/max_holding (48 barre).
L'UNICA differenza dal motore standard: se una posizione e' ancora
aperta al momento del cutoff giornaliero, viene chiusa a mercato in
quel momento (exit_reason='no_overnight'), invece di restare aperta.

BOOTSTRAP A RISCHIO FISSO (non a compounding) — correzione di metodo
del 21/07/2026, usata qui fin dall'inizio essendo un test nuovo.

Motore: nuova sottoclasse isolata di BacktestEngineFloatingKillSwitch.
SANITY CHECK: cutoff_hour=None (regola disattivata) deve produrre
risultati IDENTICI al motore standard.

NOTA: questo test NON include il risparmio/costo del funding overnight
nel calcolo del PnL della variante (il motore stesso non modella il
funding, come lo scenario "attuale lordo" usato nel conto grezzo) — sta
confrontando solo l'effetto strutturale di chiudere prima vs lasciare
aperto, sulla stessa base "lorda" di entrambi gli scenari, coerente con
come il motore ha sempre calcolato tutti i test precedenti.

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
CUTOFF_HOUR = 22  # 22:00 UTC circa, approssimazione di 22:00 UK time
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
# 1) MOTORE — sottoclasse isolata, chiusura forzata al cutoff overnight
# =====================================================================

class BacktestEngineNoOvernight(BacktestEngineFloatingKillSwitch):
    def __init__(self, *args, cutoff_hour: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cutoff_hour = cutoff_hour
        self._cutoff_tracker: dict[int, pd.Timestamp] = {}

    def _compute_first_cutoff(self, entry_time: pd.Timestamp) -> pd.Timestamp:
        fc = entry_time.normalize() + pd.Timedelta(hours=self.cutoff_hour)
        if fc < entry_time:
            fc += pd.Timedelta(days=1)
        return fc

    def _open_position_tracked(self, instrument, direction, bar, atr, adx):
        n_before = len(self.open_positions)
        self._open_position(instrument, direction, bar, atr, adx)
        if self.cutoff_hour is not None and len(self.open_positions) > n_before:
            pos = self.open_positions[-1]
            self._cutoff_tracker[id(pos)] = self._compute_first_cutoff(pos.entry_time)

    def _check_no_overnight(self, pos, bar, inst) -> bool:
        if self.cutoff_hour is None:
            return False
        cutoff = self._cutoff_tracker.get(id(pos))
        if cutoff is None or bar["timestamp"] < cutoff:
            return False
        spread = inst.spread_fixed
        exit_price = (bar["close"] - spread / 2 if pos.direction == "long"
                      else bar["close"] + spread / 2)
        self._close_position(pos, bar["timestamp"], exit_price, "no_overnight")
        self._cutoff_tracker.pop(id(pos), None)
        return True

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
                    self._cutoff_tracker.pop(id(pos), None)
                    continue

                # --- NUOVO: chiusura forzata al cutoff overnight ---
                self._check_no_overnight(pos, bar, inst)
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
                self._open_position_tracked(c["instrument"], c["direction"], c["bar"],
                                             c["atr"], c["adx"])
                slots_free -= 1

        trades_df = self.trades_to_dataframe()
        metrics_df = eng.compute_run_metrics(trades_df, self.capital0, self.capital)
        return trades_df, metrics_df


# =====================================================================
# 2) BOOTSTRAP A RISCHIO FISSO (versione corretta, usata dall'inizio qui)
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

    # --- SANITY CHECK: cutoff_hour=None deve dare risultati identici al motore standard ---
    print("=== SANITY CHECK (cutoff_hour=None deve dare risultati identici al motore standard) ===")
    p_start, p_end = pd.Timestamp("2023-01-02", tz="UTC"), pd.Timestamp("2023-12-31", tz="UTC")
    sig_check = {name: slice_period(signals_full[name], p_start, p_end) for name in SYMBOLS}

    baseline_check = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0, instruments=eng.INSTRUMENTS)
    trades_bc, _ = baseline_check.run(sig_check)
    variant_check = BacktestEngineNoOvernight(capital0=CAPITAL0, instruments=eng.INSTRUMENTS, cutoff_hour=None)
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

    # --- RUN sui 5 periodi ufficiali ---
    print("=== RUN sui 5 periodi ufficiali (bootstrap a rischio fisso) ===")
    rng = np.random.default_rng(SEED)
    rows = []
    total_deltas = np.zeros(N_BOOT)
    total_observed_delta = 0.0
    total_trades_base = 0
    total_trades_variant = 0
    total_no_overnight_exits = 0

    for label, start_str, end_str in PERIODS:
        start = pd.Timestamp(start_str, tz="UTC")
        end = pd.Timestamp(end_str, tz="UTC") + pd.Timedelta(days=1) if end_str else pd.Timestamp.now(tz="UTC")
        print(f"--- Periodo {label} ---")
        sig_period = {name: slice_period(signals_full[name], start, end) for name in SYMBOLS}

        eng_base = BacktestEngineFloatingKillSwitch(capital0=CAPITAL0, instruments=eng.INSTRUMENTS)
        trades_base, metrics_base = eng_base.run(sig_period)

        eng_variant = BacktestEngineNoOvernight(capital0=CAPITAL0, instruments=eng.INSTRUMENTS,
                                                  cutoff_hour=CUTOFF_HOUR)
        trades_variant, metrics_variant = eng_variant.run(sig_period)

        n_no_overnight = (trades_variant["exit_reason"] == "no_overnight").sum() if not trades_variant.empty else 0
        total_no_overnight_exits += n_no_overnight

        mb, mv = metrics_base.iloc[0], metrics_variant.iloc[0]
        print(f"  baseline:  {len(trades_base)} trade, pnl={mb['pnl_total']:+.2f}")
        print(f"  variante:  {len(trades_variant)} trade ({n_no_overnight} chiusi per cutoff overnight), "
              f"pnl={mv['pnl_total']:+.2f}")

        all_days = sorted(set(sig_period["DAX"]["timestamp"].dt.date) |
                           set(sig_period["FTSE100"]["timestamp"].dt.date))
        result = bootstrap_period(trades_base, trades_variant, all_days, CAPITAL0, N_BOOT, rng)
        print(f"  bootstrap: delta={result['observed_delta']:+.2f}  sd={result['boot_sd']:.2f}  "
              f"IC95%=[{result['ci_low_95']:+.2f}, {result['ci_high_95']:+.2f}]  "
              f"%iter<=0={result['pct_iter_non_positive']:.1f}%\n")

        rows.append({
            "periodo": label, "n_trade_base": len(trades_base), "n_trade_variant": len(trades_variant),
            "n_no_overnight_exits": int(n_no_overnight),
            "pnl_base": mb["pnl_total"], "pnl_variant": mv["pnl_total"],
            "delta_osservato": result["observed_delta"], "boot_sd": result["boot_sd"],
            "ci95_low": result["ci_low_95"], "ci95_high": result["ci_high_95"],
            "pct_iter_delta_non_positivo": result["pct_iter_non_positive"],
        })
        total_deltas += result["deltas"]
        total_observed_delta += result["observed_delta"]
        total_trades_base += len(trades_base)
        total_trades_variant += len(trades_variant)

    ci_low_tot, ci_high_tot = np.percentile(total_deltas, [2.5, 97.5])
    pct_non_positive_tot = (total_deltas <= 0).mean() * 100
    z_tot = total_observed_delta / total_deltas.std() if total_deltas.std() > 0 else float("nan")

    print(f"{'='*70}\nAGGREGATO — somma sui 5 periodi ufficiali (bootstrap a rischio fisso)\n{'='*70}")
    print(f"Trade totali: baseline={total_trades_base}  variante={total_trades_variant}  "
          f"(di cui {total_no_overnight_exits} chiusi per cutoff overnight)")
    print(f"Delta osservato totale (variante - baseline): {total_observed_delta:+.2f} EUR")
    print(f"Bootstrap aggregato: media={total_deltas.mean():+.2f}  sd={total_deltas.std():.2f}")
    print(f"IC 95% del delta atteso per rumore: [{ci_low_tot:+.2f}, {ci_high_tot:+.2f}]")
    print(f"% iterazioni bootstrap con delta totale <=0: {pct_non_positive_tot:.1f}%")
    print(f"z-score approssimato: {z_tot:.2f}")
    print(f"\nRicorda: questo motore NON modella il costo/risparmio del funding overnight (come tutti i "
          f"test precedenti) — misura solo l'effetto strutturale del chiudere prima vs lasciare aperto. "
          f"Il conto grezzo separato (+1.134 EUR) aveva gia' stimato l'effetto netto includendo il funding.")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/test_no_overnight.csv", index=False)
    print("\nDettaglio salvato in results/test_no_overnight.csv")


if __name__ == "__main__":
    main()
