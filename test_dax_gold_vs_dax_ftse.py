"""
test_dax_gold_vs_dax_ftse.py — Confronto sostitutivo: DAX+GOLD vs il
baseline attuale DAX+FTSE100, motore vero (BacktestEngineFloatingKillSwitch,
NESSUNA modifica al motore — solo la coppia di strumenti cambia).

Origine (24/07/2026): dopo aver verificato che il segnale V6 su GOLD
standalone e' robusto nel tempo (tutti e 5 i periodi positivi, leave-one-out
non collassa nemmeno escludendo il rally 2024-2025, +11.390EUR residui), la
domanda successiva e' se GOLD possa SOSTITUIRE FTSE100 nel pool principale
(non aggiungersi come terzo strumento — quello e' gia' stato bocciato il
22/07, z=0.24). Motivazione: corr(DAX,GOLD)=0.041 contro
corr(DAX,FTSE100)=0.72-0.93 (che sale proprio nei momenti di stress) — una
coppia DAX+GOLD offrirebbe diversificazione strutturalmente migliore,
teoricamente piu' liberta' per filtri/segnali asset-specific futuri senza
rischiare di rovinare l'altro strumento nello stesso momento di mercato.

NOTA: usa il segnale V6 STANDARD su GOLD (long+short, GOLD_CONFIG di
engine_three_asset_gold.py) — NON il filtro long-only, gia' bocciato oggi
(leave-one-out: il suo vantaggio incrementale sparisce escludendo 2024-2025,
mentre il segnale standard long+short e' robusto da solo).

Stesso protocollo di rigore usato tutto il giorno: 5 periodi ufficiali,
bootstrap a blocchi di giornata sul delta (z contro zero, non contro la
media della sua stessa distribuzione), leave-one-period-out automatico.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_three_asset_gold import GOLD_CONFIG
from ohlc_data_source import get_ohlc

CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")

PERIODS = {
    "2015-2016": ("2015-01-01", "2016-12-31"),
    "2020-covid": ("2020-01-01", "2020-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024-2025": ("2024-01-01", "2025-12-31"),
    "2026-ytd": ("2026-01-01", "2026-07-15"),
}

CAPITALE_INIZIALE = 2000.0

INSTRUMENTS_BASELINE = {
    "DAX": eng.INSTRUMENTS["DAX"],
    "FTSE100": eng.INSTRUMENTS["FTSE100"],
}
INSTRUMENTS_GOLD = {
    "DAX": eng.INSTRUMENTS["DAX"],
    "GOLD": GOLD_CONFIG,
}


def load_data():
    print("Carico OHLC DAX/FTSE100/GOLD da D1 (cache incrementale)...")
    raw = {}
    for name in ["DAX", "FTSE100", "GOLD"]:
        raw[name] = get_ohlc(name, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, log=print)
    signals = {
        "DAX": eng.generate_signals(raw["DAX"], eng.INSTRUMENTS["DAX"]),
        "FTSE100": eng.generate_signals(raw["FTSE100"], eng.INSTRUMENTS["FTSE100"]),
        "GOLD": eng.generate_signals(raw["GOLD"], GOLD_CONFIG),
    }
    return signals


def slice_period(signals: dict, start: str, end: str, keys: list[str]) -> dict:
    out = {}
    for name in keys:
        df = signals[name]
        mask = (df["timestamp"] >= pd.Timestamp(start, tz="UTC")) & (df["timestamp"] <= pd.Timestamp(end, tz="UTC"))
        out[name] = df.loc[mask].reset_index(drop=True)
    return out


def run_comparison(signals: dict):
    print("\n" + "=" * 70)
    print("CONFRONTO: DAX+FTSE100 (baseline attuale) vs DAX+GOLD (sostituzione)")
    print("=" * 70)

    results = []
    for label, (start, end) in PERIODS.items():
        data_base = slice_period(signals, start, end, ["DAX", "FTSE100"])
        data_gold = slice_period(signals, start, end, ["DAX", "GOLD"])

        eng_base = BacktestEngineFloatingKillSwitch(capital0=CAPITALE_INIZIALE, instruments=INSTRUMENTS_BASELINE)
        trades_base, _ = eng_base.run(data_base)
        pnl_base = eng_base.capital - CAPITALE_INIZIALE

        eng_gold = BacktestEngineFloatingKillSwitch(capital0=CAPITALE_INIZIALE, instruments=INSTRUMENTS_GOLD)
        trades_gold, _ = eng_gold.run(data_gold)
        pnl_gold = eng_gold.capital - CAPITALE_INIZIALE

        delta = pnl_gold - pnl_base
        print(f"\n--- {label} ---")
        print(f"  DAX+FTSE100 (baseline): {len(trades_base)} trade, PnL {pnl_base:+.2f}EUR")
        print(f"  DAX+GOLD (sostituz.):   {len(trades_gold)} trade, PnL {pnl_gold:+.2f}EUR")
        print(f"  Delta: {delta:+.2f}EUR")

        results.append({
            "periodo": label, "pnl_base": pnl_base, "pnl_gold": pnl_gold, "delta": delta,
            "trades_base": trades_base, "trades_gold": trades_gold,
        })

    return results


def bootstrap_and_diagnostics(results: list, n_iter: int = 2000, seed: int = 42):
    print("\n" + "=" * 70)
    print("BOOTSTRAP a blocchi di giornata + leave-one-period-out")
    print("=" * 70)

    rng = np.random.default_rng(seed)
    all_days_base, all_days_gold = {}, {}
    for r in results:
        tb, tg = r["trades_base"], r["trades_gold"]
        if len(tb):
            tb = tb.copy(); tb["day"] = pd.to_datetime(tb["entry_time"]).dt.date.astype(str)
        if len(tg):
            tg = tg.copy(); tg["day"] = pd.to_datetime(tg["entry_time"]).dt.date.astype(str)
        for day, grp in (tb.groupby("day") if len(tb) else []):
            all_days_base.setdefault(day, []).extend(grp["pnl"].tolist())
        for day, grp in (tg.groupby("day") if len(tg) else []):
            all_days_gold.setdefault(day, []).extend(grp["pnl"].tolist())

    days = sorted(set(all_days_base) | set(all_days_gold))
    observed_delta = sum(sum(v) for v in all_days_gold.values()) - sum(sum(v) for v in all_days_base.values())

    null_deltas = []
    for _ in range(n_iter):
        sampled_days = rng.choice(days, size=len(days), replace=True)
        d_base = sum(sum(all_days_base.get(d, [])) for d in sampled_days)
        d_gold = sum(sum(all_days_gold.get(d, [])) for d in sampled_days)
        null_deltas.append(d_gold - d_base)
    null_deltas = np.array(null_deltas)

    z = observed_delta / null_deltas.std()
    ci_low, ci_high = np.percentile(null_deltas, [2.5, 97.5])
    pct_flip = (null_deltas <= 0).mean() if observed_delta > 0 else (null_deltas >= 0).mean()

    print(f"\nDelta osservato (DAX+GOLD - DAX+FTSE100), aggregato 5 periodi: {observed_delta:+.2f} EUR")
    print(f"IC 95% bootstrap: [{ci_low:+.2f}, {ci_high:+.2f}] EUR")
    print(f"Z-score (contro delta=0): {z:.3f}")
    print(f"Frazione iterazioni bootstrap con segno opposto/nullo: {pct_flip*100:.1f}%")

    print("\n--- Controllo leave-one-period-out ---")
    concentrazione_sospetta = False
    for r in results:
        resto = observed_delta - r["delta"]
        segno_cambia = (observed_delta > 0) != (resto > 0) if abs(resto) > 0.01 else True
        flag = "  <-- ATTENZIONE" if segno_cambia else ""
        print(f"  Escludendo {r['periodo']}: delta residuo = {resto:+.2f} EUR{flag}")
        if segno_cambia:
            concentrazione_sospetta = True

    if concentrazione_sospetta:
        verdetto = "NON PROMOSSO — dipende da un solo periodo"
    elif abs(z) >= 2 and (ci_low > 0 or ci_high < 0):
        verdetto = "PROMOSSO — DAX+GOLD batte DAX+FTSE100 in modo robusto"
    else:
        verdetto = "NON PROMOSSO / AMBIGUO"
    print(f"\nVerdetto: {verdetto}")


def main():
    if not CLOUDFLARE_ACCOUNT_ID or not CLOUDFLARE_API_TOKEN:
        print("ERRORE: CLOUDFLARE_ACCOUNT_ID o CLOUDFLARE_API_TOKEN mancanti.")
        return
    signals = load_data()
    results = run_comparison(signals)
    bootstrap_and_diagnostics(results)


if __name__ == "__main__":
    main()
