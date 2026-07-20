"""
bootstrap_gold_significance.py — Bootstrap noise-check (20/07/2026) sul
vantaggio aggregato di V6+GOLD (selezione a correlazione pura) sui 5
periodi ufficiali, +2.582,77EUR (engine_three_asset_gold_compare_test.py,
19/07/2026).

Domanda: quel numero supera chiaramente il rumore campionario atteso, o
e' dentro l'incertezza attesa per puro caso (Protocollo Anti-Rumore,
principio 4)?

METODO — bootstrap a blocchi di giornata, size dinamica via R-multiple,
kill switch riapplicato (Regole_Backtest_MonteCarlo.md sez.5, MAI la
versione naive che rimescola i singoli trade senza questi vincoli).

Per ciascuno dei 5 periodi ufficiali:
  1. Rigioca il motore REALE (BacktestEngineFloatingKillSwitch baseline
     vs BacktestEngineV6Gold) per ottenere i trade veri (selezione
     candidati/concorrenza/correlazione gia' avvenuta nel motore).
  2. Raggruppa i trade per giorno di calendario (entry_time.date()).
  3. Bootstrap: ricampiona con reinserimento i GIORNI dell'intero
     periodo (blocco = 1 giorno di calendario, non il singolo trade).
     STESSA sequenza di giorni ricampionati applicata in parallelo a
     baseline e +GOLD (confronto appaiato — isola l'effetto di GOLD
     dalla varianza generica del periodo).
  4. Per ogni iterazione, rigioca la sequenza ricampionata su un pool di
     capitale che parte da CAPITAL_V6, ricalcolando ogni trade come
     pnl = r_multiple * capitale_corrente * risk_pct_strumento (size
     dinamica) e riapplicando il kill switch giornaliero -4% (se una
     giornata sintetica supera la soglia, i trade successivi di quella
     stessa giornata vengono saltati, come nel motore reale).
  5. Il delta (+GOLD - baseline) per iterazione forma la distribuzione
     di rumore atteso. Il delta osservato (path reale, non ricampionato)
     viene confrontato contro questa distribuzione (percentile, z-score
     approssimato, % iterazioni con segno opposto).

APPROSSIMAZIONE DICHIARATA (da non nascondere nei risultati): il replay
usa i trade GIA' selezionati dal motore originale (con la loro size
reale via risk_pct statico per strumento) — non riesegue la selezione
candidati bar-per-bar sul path ricampionato. E' lo stesso livello di
approssimazione descritto in Regole_Backtest_MonteCarlo.md sez.5
("size dinamica via R-multiple"), non una riproduzione esatta bar-by-bar
del motore. Trade con size forzata al minimo (forced_min_size=True)
sono replicati con la stessa formula (piccola imprecisione nota, non
corretta qui — segnalata, non nascosta).

Output: SOLO aggregati per periodo + totale (nessun trade individuale,
nessun path-by-path) — coerente con Regole_Backtest_MonteCarlo.md sez.2-3.
Nessuna scrittura su D1. Nessuna modifica a engine.py o alle sottoclassi.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import engine as eng
from engine_floating_kill_switch import BacktestEngineFloatingKillSwitch
from engine_three_asset_gold import BacktestEngineV6Gold, instruments_with_gold, GOLD_CONFIG
from ohlc_data_source import get_ohlc

CAPITAL_V6 = 1400.0
SYMBOLS_3 = ["DAX", "FTSE100", "GOLD"]
N_BOOT = 2000
SEED = 20260720
KILL_SWITCH_PCT = eng.PARAMS.kill_switch_pct  # -4%, stesso valore del motore

PERIODS = [
    ("2015-2016", "2015-01-05", "2016-12-29"),
    ("2020-covid", "2020-01-02", "2020-12-30"),
    ("2023", "2023-01-02", "2023-12-30"),
    ("2024-2025", "2024-01-03", "2025-12-31"),
    ("2026-ytd", "2026-01-05", "2026-07-10"),
]

# risk_pct statico per strumento (usato per la size dinamica nel replay,
# stesso valore usato dal motore reale in _position_size)
RISK_PCT_BY_INSTRUMENT = {
    "DAX": eng.INSTRUMENTS["DAX"].risk_pct,
    "FTSE100": eng.INSTRUMENTS["FTSE100"].risk_pct,
    "GOLD": GOLD_CONFIG.risk_pct,
}


def slice_period(df: pd.DataFrame, p_start: pd.Timestamp, p_end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["timestamp"] >= p_start) & (df["timestamp"] < p_end)].reset_index(drop=True)


def build_day_index(trades_df: pd.DataFrame) -> dict:
    """{giorno: [(risk_pct, r_multiple), ...]} in ordine cronologico intraday."""
    if trades_df.empty:
        return {}
    df = trades_df.copy()
    df["entry_day"] = pd.to_datetime(df["entry_time"]).dt.date
    df["risk_pct_used"] = df["instrument"].map(RISK_PCT_BY_INSTRUMENT)
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


def bootstrap_period(base_trades: pd.DataFrame, gold_trades: pd.DataFrame,
                      all_days: list, capital0: float, n_boot: int,
                      rng: np.random.Generator) -> dict:
    base_idx = build_day_index(base_trades)
    gold_idx = build_day_index(gold_trades)
    n_days = len(all_days)

    observed_base = replay_path(all_days, base_idx, capital0)
    observed_gold = replay_path(all_days, gold_idx, capital0)
    observed_delta = observed_gold - observed_base

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        sampled_days = [all_days[i] for i in rng.integers(0, n_days, size=n_days)]
        pnl_base = replay_path(sampled_days, base_idx, capital0)
        pnl_gold = replay_path(sampled_days, gold_idx, capital0)
        deltas[b] = pnl_gold - pnl_base

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    pct_non_positive = (deltas <= 0).mean() * 100
    z = observed_delta / deltas.std() if deltas.std() > 0 else float("nan")

    return {
        "observed_base_pnl": observed_base,
        "observed_gold_pnl": observed_gold,
        "observed_delta": observed_delta,
        "boot_mean": deltas.mean(),
        "boot_sd": deltas.std(),
        "ci_low_95": ci_low,
        "ci_high_95": ci_high,
        "pct_iter_non_positive": pct_non_positive,
        "z_approx": z,
        "deltas": deltas,  # tenuto solo in memoria per l'aggregazione, MAI scritto su file
    }


def main():
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=== Bootstrap noise-check — vantaggio V6+GOLD (correlazione pura), 5 periodi ufficiali ===")
    log(f"N_BOOT={N_BOOT} per periodo, seed={SEED}, capitale pool V6={CAPITAL_V6}\n")

    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        log("ERRORE: CLOUDFLARE_API_TOKEN o CLOUDFLARE_ACCOUNT_ID mancanti.")
        return

    instruments_2 = dict(eng.INSTRUMENTS)
    instruments_3 = instruments_with_gold()

    log("Verifico/aggiorno OHLC (D1 + eventuali barre mancanti da Dukascopy)...")
    raw_full = {name: get_ohlc(name, account_id, token, log=log) for name in SYMBOLS_3}
    log("Fatto.\n")

    v6_signals_full = {name: eng.generate_signals(raw_full[name], instruments_3[name]) for name in SYMBOLS_3}

    rng = np.random.default_rng(SEED)
    rows = []
    total_deltas = np.zeros(N_BOOT)
    total_observed_delta = 0.0

    for label, p_start_str, p_end_str in PERIODS:
        p_start = pd.Timestamp(p_start_str, tz="UTC")
        p_end = pd.Timestamp(p_end_str, tz="UTC") + pd.Timedelta(days=1)
        log(f"--- Periodo {label} ---")

        sig_3 = {name: slice_period(v6_signals_full[name], p_start, p_end) for name in SYMBOLS_3}
        sig_2 = {k: v for k, v in sig_3.items() if k != "GOLD"}

        eng_base = BacktestEngineFloatingKillSwitch(capital0=CAPITAL_V6, instruments=instruments_2)
        trades_base, _ = eng_base.run(sig_2)

        eng_gold = BacktestEngineV6Gold(capital0=CAPITAL_V6, instruments=instruments_3)
        trades_gold, _ = eng_gold.run(sig_3)

        all_days = sorted(set(sig_3["DAX"]["timestamp"].dt.date) |
                           set(sig_3["FTSE100"]["timestamp"].dt.date))

        result = bootstrap_period(trades_base, trades_gold, all_days, CAPITAL_V6, N_BOOT, rng)

        log(f"  Trade: baseline={len(trades_base)}  +GOLD={len(trades_gold)}  giorni periodo={len(all_days)}")
        log(f"  PnL osservato: baseline={result['observed_base_pnl']:+.2f}  "
            f"+GOLD={result['observed_gold_pnl']:+.2f}  delta={result['observed_delta']:+.2f}")
        log(f"  Bootstrap ({N_BOOT} iter, blocchi di giornata): media={result['boot_mean']:+.2f}  "
            f"sd={result['boot_sd']:.2f}")
        log(f"  IC 95% del delta atteso per rumore: [{result['ci_low_95']:+.2f}, {result['ci_high_95']:+.2f}]")
        log(f"  % iterazioni bootstrap con delta<=0: {result['pct_iter_non_positive']:.1f}%")
        log(f"  z-score approssimato (delta osservato / sd bootstrap): {result['z_approx']:.2f}\n")

        rows.append({
            "periodo": label,
            "n_trade_base": len(trades_base), "n_trade_gold": len(trades_gold),
            "n_giorni": len(all_days),
            "delta_osservato": result["observed_delta"],
            "boot_media": result["boot_mean"], "boot_sd": result["boot_sd"],
            "ci95_low": result["ci_low_95"], "ci95_high": result["ci_high_95"],
            "pct_iter_delta_non_positivo": result["pct_iter_non_positive"],
            "z_approx": result["z_approx"],
        })

        total_deltas += result["deltas"]
        total_observed_delta += result["observed_delta"]

    # ============================================================
    # Aggregato sui 5 periodi (somma dei delta per iterazione — i
    # periodi sono blocchi storici indipendenti, coerente col resto
    # del protocollo walk-forward del progetto)
    # ============================================================
    ci_low_tot, ci_high_tot = np.percentile(total_deltas, [2.5, 97.5])
    pct_non_positive_tot = (total_deltas <= 0).mean() * 100
    z_tot = total_observed_delta / total_deltas.std() if total_deltas.std() > 0 else float("nan")

    log(f"{'='*70}\nAGGREGATO — somma sui 5 periodi ufficiali\n{'='*70}")
    log(f"Delta osservato totale (motore reale, non ricampionato): {total_observed_delta:+.2f}")
    log(f"  (per confronto: engine_three_asset_gold_compare_test.py 19/07/2026 = +2.582,77)")
    log(f"Bootstrap aggregato: media={total_deltas.mean():+.2f}  sd={total_deltas.std():.2f}")
    log(f"IC 95% del delta atteso per rumore: [{ci_low_tot:+.2f}, {ci_high_tot:+.2f}]")
    log(f"% iterazioni bootstrap con delta totale <=0: {pct_non_positive_tot:.1f}%")
    log(f"z-score approssimato: {z_tot:.2f}")
    log(f"\nInterpretazione: se il delta osservato (+{total_observed_delta:.0f}EUR) cade DENTRO "
        f"l'IC 95% del rumore atteso, o se la %% di iterazioni <=0 e' alta (es. >10-15%%), "
        f"il vantaggio GOLD non e' chiaramente distinguibile dal rumore campionario del "
        f"periodo storico — coerente col Protocollo Anti-Rumore principio 4/7.")

    summary_df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv("results/bootstrap_gold_significance.csv", index=False)
    with open("results/bootstrap_gold_significance.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\n=== Completato. Output SOLO aggregato in results/. ===")


if __name__ == "__main__":
    main()
