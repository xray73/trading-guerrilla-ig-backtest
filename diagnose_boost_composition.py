"""
diagnose_boost_composition.py — Verifica se la progressione di z nello
sweep (test_ftse_composite_boost_sweep.py) è guidata da un edge che
scala linearmente col moltiplicatore, o da effetti non lineari (size
forzata al minimo tradabile, riduzione per margine) che cambiano la
composizione dei trade tra un moltiplicatore e l'altro.

Se un moltiplicatore basso (es. 1.1x) forza più trade al minimo
tradabile rispetto a uno alto (perché risk_amount più basso cade più
spesso sotto la soglia), quei trade NON scalano affatto col
moltiplicatore — la loro size è fissa (0.50), indipendente da quanto
rischio "vorrebbero" usare. Questo significa che l'incremento di z tra
moltiplicatori bassi e alti potrebbe derivare in parte dal fatto che
più trade "sbloccano" size piena a moltiplicatori più alti, non da un
edge che genuinamente migliora con più rischio.

Stampa, per ciascuno dei 4 moltiplicatori giA testati, SOLO per i
trade FTSE100 in stato stress (quelli davvero toccati dal boost):
  - n trade totali
  - n forzati al minimo tradabile (forced_min_size=True)
  - % forzati al minimo
  - n con margine ridotto (dal contatore aggregato del motore)

Nessuna scrittura su D1, solo stampa aggregata.
"""
import os
import pandas as pd
import requests

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from ohlc_data_source import get_ohlc

CF_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
CF_API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]
D1_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"

CAPITAL_V6 = 1400.0
MULTIPLIERS_TO_TEST = [1.1, 1.3, 1.5, 1.7]
ATR_THRESHOLD_FTSE = 0.2031323100223204
CORR_THRESHOLD = 0.7853464827260775
TARGET_INSTRUMENT = "FTSE100"

PERIODS = {
    "2015-2016": ("2015-01-01", "2016-12-31"),
    "2020-covid": ("2020-01-01", "2020-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024-2025": ("2024-01-01", "2025-12-31"),
    "2026-ytd": ("2026-01-01", "2026-07-12"),
}


def d1(sql):
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{D1_ID}/query"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    resp = requests.post(url, json={"sql": sql}, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(data.get("errors"))
    return data["result"][0]["results"]


class BacktestEngineFtseCompositeBoost(BacktestEngineFloatingKillSwitch):
    def __init__(self, capital0, stress_lookup: pd.Series, boost_multiplier: float, **kwargs):
        super().__init__(capital0, **kwargs)
        self.stress_lookup = stress_lookup
        self.boost_multiplier = boost_multiplier
        self._current_instrument = None
        self._current_entry_time = None
        self._stress_flags = []  # traccia se il trade N-esimo aperto era in stress FTSE100

    def _is_stress(self, ts) -> bool:
        if self.stress_lookup.empty:
            return False
        idx = self.stress_lookup.index.searchsorted(ts, side="right") - 1
        if idx < 0:
            return False
        return bool(self.stress_lookup.iloc[idx])

    def _open_position(self, instrument, direction, bar, atr_at_entry, adx_at_entry):
        self._current_instrument = instrument
        self._current_entry_time = bar["timestamp"]
        is_stress_trade = (instrument == TARGET_INSTRUMENT and self._is_stress(bar["timestamp"]))
        n_before = len(self.closed_trades) + len(self.open_positions)
        super()._open_position(instrument, direction, bar, atr_at_entry, adx_at_entry)
        n_after = len(self.closed_trades) + len(self.open_positions)
        if n_after > n_before:  # una posizione e' stata davvero aperta (size>0)
            self._stress_flags.append(is_stress_trade)

    def _position_size(self, entry_price, stop_price, inst):
        risk_amount = self.capital * inst.risk_pct

        multiplier = 1.0
        if self._current_instrument == TARGET_INSTRUMENT and self._is_stress(self._current_entry_time):
            multiplier = self.boost_multiplier
        risk_amount *= multiplier

        risk_distance = abs(entry_price - stop_price)
        if risk_distance <= 0:
            return 0.0, 0.0, False, False

        size = risk_amount / (risk_distance * inst.point_value)
        forced_min_size = False
        if size < inst.min_tradable_size:
            size = inst.min_tradable_size
            forced_min_size = True

        margin_required = size * entry_price * inst.point_value * inst.margin_pct
        margin_reduced = False
        if margin_required > self.capital:
            max_size_by_margin = self.capital / (entry_price * inst.point_value * inst.margin_pct)
            if max_size_by_margin < size:
                size = max(max_size_by_margin, 0.0)
                margin_reduced = True

        return size, risk_amount, forced_min_size, margin_reduced


def slice_period(signals, start, end):
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return signals[(signals["timestamp"] >= start_ts) & (signals["timestamp"] < end_ts)].reset_index(drop=True)


def main():
    print("Scarico OHLC continuo 2015-2026 (DAX+FTSE100)...")
    hist = {name: get_ohlc(name, CF_ACCOUNT_ID, CF_API_TOKEN) for name in ("DAX", "FTSE100")}

    print("Genero segnali V6...")
    signals = {name: eng.generate_signals(hist[name], eng.INSTRUMENTS[name]) for name in hist}

    print("Scarico atr_pct (FTSE100) e corr_dax_ftse_7d...")
    rows_atr = d1(f"SELECT timestamp, atr_pct FROM market_regime_indicators "
                  f"WHERE instrument='{TARGET_INSTRUMENT}' AND atr_pct IS NOT NULL ORDER BY timestamp ASC")
    rows_corr = d1("SELECT timestamp, corr_dax_ftse_7d FROM market_regime_indicators "
                   "WHERE instrument='DAX' AND corr_dax_ftse_7d IS NOT NULL ORDER BY timestamp ASC")

    atr_series = pd.Series(
        [r["atr_pct"] for r in rows_atr],
        index=pd.to_datetime([r["timestamp"] for r in rows_atr], utc=True))
    corr_series = pd.Series(
        [r["corr_dax_ftse_7d"] for r in rows_corr],
        index=pd.to_datetime([r["timestamp"] for r in rows_corr], utc=True))

    combined = pd.concat([atr_series.rename("atr"), corr_series.rename("corr")], axis=1, sort=True).dropna()
    stress_lookup = (combined["atr"] > ATR_THRESHOLD_FTSE) & (combined["corr"] > CORR_THRESHOLD)

    print(f"\n{'Molt.':<8}{'N trade FTSE/stress':>22}{'N forzati al minimo':>22}{'%% forzati':>12}{'N margine ridotto':>20}")

    for m in MULTIPLIERS_TO_TEST:
        all_trades = []
        total_margin_reduced = 0
        for period_name, (start, end) in PERIODS.items():
            sliced = {name: slice_period(sig, start, end) for name, sig in signals.items()}
            engine_ = BacktestEngineFtseCompositeBoost(
                capital0=CAPITAL_V6, stress_lookup=stress_lookup, boost_multiplier=m)
            trades_df, _ = engine_.run(sliced)
            total_margin_reduced += engine_.n_margin_reduced
            if not trades_df.empty:
                all_trades.append(trades_df)

        combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

        if combined_trades.empty:
            print(f"{m:<8}{'0':>22}{'0':>22}{'n/d':>12}{total_margin_reduced:>20}")
            continue

        # trade FTSE100 il cui entry_time cade in stato stress
        is_ftse = combined_trades["instrument"] == TARGET_INSTRUMENT
        entry_ts = pd.to_datetime(combined_trades["entry_time"], utc=True)
        idx_stress = entry_ts.apply(lambda ts: bool(stress_lookup.iloc[
            max(stress_lookup.index.searchsorted(ts, side="right") - 1, 0)
        ]) if stress_lookup.index.searchsorted(ts, side="right") - 1 >= 0 else False)
        stress_trades = combined_trades[is_ftse & idx_stress]

        n_stress = len(stress_trades)
        n_forced = int(stress_trades["forced_min_size"].sum()) if n_stress else 0
        pct_forced = 100 * n_forced / n_stress if n_stress else 0.0

        print(f"{m:<8}{n_stress:>22}{n_forced:>22}{pct_forced:>11.1f}%{total_margin_reduced:>20}")

    print("\nInterpretazione: se '%% forzati' SCENDE parecchio da 1.1x a 1.7x, parte della "
          "crescita di z nello sweep e' meccanica (piu' trade 'sbloccano' size piena a "
          "moltiplicatori alti), non solo edge che scala col rischio. Se resta stabile, "
          "la progressione di z e' piu' probabilmente genuina.")


if __name__ == "__main__":
    main()
