"""
index_pipeline.py — Pipeline unica per validare un nuovo indice candidato,
dall'inizio alla fine, in un solo run. Consolida in un unico script tutti
i passi finora fatti manualmente uno alla volta (discover_dukascopy_*.py,
load_ohlc_generic.py, correlazione SQL, persistence_check_generic.py,
single_instrument_validation.py, smi_atr_lookback_grid.py).

Uso: python index_pipeline.py SIMBOLO [--reference FTSE100]

FASE 0 — Scoperta strumento Dukascopy (con freno di sicurezza)
  Se il simbolo NON è già in INSTRUMENT_REGISTRY (parametri IG
  verificati via screenshot: size minima, margine, point_value, spread)
  E la costante Dukascopy non è già in SYMBOL_DUKASCOPY_MAP, la pipeline
  cerca automaticamente tra le costanti IDX/INDICES — ma se trova ZERO
  o PIÙ DI UN candidato ambiguo, SI FERMA e stampa i candidati invece
  di indovinare. Lezione imparata da SMI: la prima ricerca automatica
  ha trovato solo azioni svizzere singole (falso positivo pericoloso),
  serve conferma umana prima di scaricare dati sotto un nome sbagliato.

  Per procedere dopo una scoperta ambigua: aggiungere manualmente la
  costante confermata a SYMBOL_DUKASCOPY_MAP e i parametri IG (da
  screenshot "Get Info") a INSTRUMENT_REGISTRY, poi rilanciare — da
  quel momento il simbolo passa in modalità completamente automatica.

FASE 1 — Download OHLC (5 periodi standard) via Dukascopy
FASE 2 — Caricamento in D1 (idempotente: cancella e ricarica), via
  HTTP diretto all'API D1 (nessun wrangler/Node necessario in questo
  script — se il volume di righe è molto alto, i batch INSERT sono
  comunque spezzati in chunk per restare sotto i limiti D1)
FASE 3 — Correlazione vs DAX e FTSE100 (rendimenti giornalieri, calcolo
  diretto in pandas sui dati già scaricati in FASE 1, nessuna nuova
  query D1 necessaria)
FASE 4 — Persistenza direzionale grezza vs simbolo di riferimento
  (default FTSE100, configurabile con --reference)
FASE 5 — Validazione statistica segnale (z-score vs 30 baseline random
  per periodo, parametri IG grezzi/non calibrati)
FASE 6 — Calibrazione ATR×lookback (grid search train 2023, verifica
  sui 4 periodi restanti, confronto vs parametri grezzi di FASE 5)

Ogni fase stampa il proprio risultato e viene salvata in results/. Il
report finale riassume tutte le fasi con un verdetto complessivo, ma
NON prende decisioni al posto tuo — resta a te decidere se procedere
al confronto di coppia (asset_pair_comparison.py) dopo aver letto i
numeri.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import dataclasses

import numpy as np
import pandas as pd
import requests

import engine as eng

DATABASE_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CHUNK_SIZE_WRITE = 400
CHUNK_SIZE_READ = 5000

PERIODS = {
    "2015-2016": ("2015-01-01", "2016-12-31"),
    "2020-covid": ("2020-01-01", "2020-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024-2025": ("2024-01-01", "2025-12-31"),
    "2026-ytd": ("2026-01-01", "2026-07-14"),
}
WARMUP_DAYS = 90
TRAIN_PERIOD = "2023"
VERIFY_PERIODS = ["2015-2016", "2020-covid", "2024-2025", "2026-ytd"]

# ── Costanti Dukascopy già confermate (discovery fatta una volta, riusabile) ──
SYMBOL_DUKASCOPY_MAP = {
    "ITALY40": "INSTRUMENT_IDX_EUROPE_ITA_IDX_EUR",
    "SMI": "INSTRUMENT_IDX_EUROPE_E_SWMI",
    "IBEX35": "INSTRUMENT_IDX_EUROPE_E_IBC_MAC",  # non verificata al 100%, vedi load_ohlc_generic.py
}

# ── Parametri IG verificati via screenshot "Get Info" (size/margine/spread) ──
INSTRUMENT_REGISTRY: dict[str, eng.InstrumentConfig] = {
    "SMI": eng.InstrumentConfig(
        name="SMI", tradable=True,
        breakout_lookback=20, atr_multiplier=1.5, risk_pct=0.015,
        point_value=2.16, spread_fixed=2.0,  # spread verificato mercato attivo 15/07/2026
        min_tradable_size=0.10, margin_pct=0.10,
    ),
    # TOPIX: parametri da screenshot 15/07/2026 (size/margine) + spread
    # verificato in orario di mercato attivo nello stesso giorno.
    # Costante Dukascopy NON ancora confermata — vedi FASE 0.
    "TOPIX": eng.InstrumentConfig(
        name="TOPIX", tradable=True,
        breakout_lookback=20, atr_multiplier=1.5, risk_pct=0.015,  # grezzi, da calibrare
        point_value=0.87, spread_fixed=0.8,
        min_tradable_size=0.50, margin_pct=0.10,
    ),
}


# ══════════════════════════════════════════════════════════════════
# FASE 0 — Scoperta / conferma strumento Dukascopy
# ══════════════════════════════════════════════════════════════════

def phase0_resolve_dukascopy_constant(symbol: str) -> str:
    if symbol in SYMBOL_DUKASCOPY_MAP:
        const_name = SYMBOL_DUKASCOPY_MAP[symbol]
        print(f"[FASE 0] Costante già confermata: {symbol} -> {const_name}")
        return const_name

    print(f"[FASE 0] Costante per {symbol} non ancora confermata. Cerco tra le "
          f"costanti indice disponibili in dukascopy-python...")
    import dukascopy_python.instruments as instr
    all_attrs = [a for a in dir(instr) if a.startswith("INSTRUMENT_")]
    idx_attrs = [a for a in all_attrs if "IDX" in a.upper() or "INDICES" in a.upper()]

    # euristica di ricerca: nome simbolo stesso + alias comuni noti
    aliases = {
        "TOPIX": ["TOPIX", "TOKYO", "JPN", "JAPAN"],
        "IBEX35": ["IBEX", "SPAIN", "MAC", "IBC"],
    }
    search_terms = aliases.get(symbol, [symbol])

    candidates = []
    for term in search_terms:
        candidates.extend([a for a in idx_attrs if term.upper() in a.upper()])
    candidates = sorted(set(candidates))

    if len(candidates) == 1:
        const_name = candidates[0]
        print(f"[FASE 0] Un solo candidato trovato: {const_name} = "
              f"{getattr(instr, const_name)!r}. Procedo, ma NON è stato "
              f"confermato da un discover dedicato in precedenza — verifica "
              f"il conteggio righe/range date dopo il download prima di fidarti.")
        return const_name

    print(f"\n[FASE 0] STOP — trovati {len(candidates)} candidati (serve 1 solo):")
    for c in candidates:
        print(f"    {c} = {getattr(instr, c)!r}")
    if not candidates:
        print("    Nessun candidato trovato con gli alias attuali. Elenco "
              "completo costanti indice per ricerca manuale:")
        for a in sorted(idx_attrs):
            print(f"    {a} = {getattr(instr, a)!r}")
    print(f"\nAggiungi manualmente la costante corretta a SYMBOL_DUKASCOPY_MAP "
          f"in questo script, poi rilancia. Interrompo qui — non indovino.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════
# FASE 1 — Download OHLC
# ══════════════════════════════════════════════════════════════════

def phase1_download_ohlc(symbol: str, dukascopy_const_name: str) -> pd.DataFrame:
    import dukascopy_python
    import dukascopy_python.instruments as instr
    from datetime import datetime

    instrument = getattr(instr, dukascopy_const_name)
    print(f"\n[FASE 1] Download OHLC {symbol} (costante: {dukascopy_const_name})...")

    all_rows = []
    for label, (start_str, end_str) in PERIODS.items():
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        df = dukascopy_python.fetch(
            instrument, dukascopy_python.INTERVAL_MIN_30,
            dukascopy_python.OFFER_SIDE_BID, start, end,
        ).reset_index()
        n = len(df)
        print(f"  {label}: {n} barre")
        if n == 0:
            print(f"    ATTENZIONE: zero barre per {label}.")
            continue
        ts_col = df.columns[0]
        df = df.rename(columns={ts_col: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        all_rows.append(df[["timestamp", "open", "high", "low", "close", "volume"]])

    if not all_rows:
        print("[FASE 1] ERRORE: nessuna barra scaricata in nessun periodo. Interrompo.")
        sys.exit(1)

    full_df = pd.concat(all_rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    full_df = full_df.drop_duplicates(subset="timestamp")
    print(f"[FASE 1] Totale barre: {len(full_df)}")

    os.makedirs("results", exist_ok=True)
    full_df.to_csv(f"{symbol}_full.csv", index=False)
    return full_df


# ══════════════════════════════════════════════════════════════════
# FASE 2 — Caricamento D1 (via HTTP diretto, no wrangler)
# ══════════════════════════════════════════════════════════════════

def d1_query(sql: str, account_id: str, token: str) -> dict:
    url = f"{API_BASE}/{account_id}/d1/database/{DATABASE_ID}/query"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"sql": sql}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"D1 query fallita: {data.get('errors')}")
    return data


def phase2_load_d1(symbol: str, full_df: pd.DataFrame, account_id: str, token: str):
    print(f"\n[FASE 2] Caricamento {symbol} in D1 (idempotente)...")
    d1_query(f"DELETE FROM ohlc_prices WHERE symbol='{symbol}';", account_id, token)

    rows = list(full_df.itertuples(index=False))
    n_chunks = (len(rows) + CHUNK_SIZE_WRITE - 1) // CHUNK_SIZE_WRITE
    for i in range(0, len(rows), CHUNK_SIZE_WRITE):
        chunk = rows[i:i + CHUNK_SIZE_WRITE]
        values = ",\n".join(
            f"('{symbol}', '{r.timestamp}', '30m', {r.open}, {r.high}, {r.low}, {r.close}, "
            f"{r.volume if pd.notna(r.volume) else 0}, 'dukascopy')"
            for r in chunk
        )
        sql = (f"INSERT INTO ohlc_prices (symbol, timestamp, timeframe, open, high, low, "
               f"close, volume, source) VALUES\n{values};")
        d1_query(sql, account_id, token)
        if (i // CHUNK_SIZE_WRITE) % 20 == 0:
            print(f"  chunk {i // CHUNK_SIZE_WRITE + 1}/{n_chunks}...")
        time.sleep(0.1)

    result = d1_query(f"SELECT COUNT(*) as n, MIN(timestamp) as mn, MAX(timestamp) as mx "
                       f"FROM ohlc_prices WHERE symbol='{symbol}';", account_id, token)
    row = result["result"][0]["results"][0]
    print(f"[FASE 2] Verifica: {row['n']} righe caricate, range {row['mn']} -> {row['mx']}")


# ══════════════════════════════════════════════════════════════════
# FASE 3 — Correlazione (diretta in pandas, dati già in memoria)
# ══════════════════════════════════════════════════════════════════

def fetch_reference_daily_closes(symbol: str, account_id: str, token: str) -> pd.Series:
    rows, offset = [], 0
    while True:
        sql = (f"SELECT timestamp, close FROM ohlc_prices WHERE symbol='{symbol}' "
               f"ORDER BY timestamp LIMIT {CHUNK_SIZE_READ} OFFSET {offset}")
        batch = d1_query(sql, account_id, token)["result"][0]["results"]
        if not batch:
            break
        rows.extend(batch)
        offset += CHUNK_SIZE_READ
        if len(batch) < CHUNK_SIZE_READ:
            break
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["d"] = df["timestamp"].dt.date
    daily = df.groupby("d")["close"].last()
    return daily


def phase3_correlation(symbol: str, full_df: pd.DataFrame, account_id: str, token: str) -> dict:
    print(f"\n[FASE 3] Correlazione {symbol} vs DAX/FTSE100...")
    target = full_df.copy()
    target["d"] = target["timestamp"].dt.date
    target_daily = target.groupby("d")["close"].last()
    target_ret = target_daily.pct_change().dropna()

    results = {}
    for ref_symbol in ["DAX", "FTSE100"]:
        ref_daily = fetch_reference_daily_closes(ref_symbol, account_id, token)
        ref_ret = ref_daily.pct_change().dropna()
        joined = pd.DataFrame({"target": target_ret, "ref": ref_ret}).dropna()
        corr = joined["target"].corr(joined["ref"]) if len(joined) > 2 else np.nan
        results[f"corr_{symbol}_{ref_symbol}"] = corr
        print(f"  {symbol}-{ref_symbol}: {corr:.3f} (n={len(joined)} giorni)")

    return results


# ══════════════════════════════════════════════════════════════════
# FASE 4 — Persistenza direzionale grezza
# ══════════════════════════════════════════════════════════════════

def compute_persistence(df: pd.DataFrame, forward_bars: int = 20) -> dict:
    out = df.copy()
    out["ema20"] = eng.ema(out["close"], 20)
    out["ema50"] = eng.ema(out["close"], 50)
    out["adx"] = eng.adx_wilder(out, 14)

    direction_long = out["ema20"] > out["ema50"]
    direction_short = out["ema20"] < out["ema50"]
    trend_context = out["adx"] > 20

    out["forward_close"] = out["close"].shift(-forward_bars)
    out["forward_return"] = (out["forward_close"] - out["close"]) / out["close"]

    long_ctx = out[trend_context & direction_long & out["forward_return"].notna()]
    short_ctx = out[trend_context & direction_short & out["forward_return"].notna()]

    n_total_ctx = len(long_ctx) + len(short_ctx)
    if n_total_ctx > 0:
        combined = ((long_ctx["forward_return"] > 0).sum() +
                    (short_ctx["forward_return"] < 0).sum()) / n_total_ctx
    else:
        combined = np.nan
    return {"n_contesto_trend": n_total_ctx, "persistenza_combinata": combined}


def slice_period(df: pd.DataFrame, period_label: str):
    start_str, end_str = PERIODS[period_label]
    start = pd.Timestamp(start_str, tz="UTC") - pd.Timedelta(days=WARMUP_DAYS)
    end = pd.Timestamp(end_str, tz="UTC") + pd.Timedelta(days=1)
    window = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
    return window, pd.Timestamp(start_str, tz="UTC")


def trim_warmup(df: pd.DataFrame, period_start: pd.Timestamp) -> pd.DataFrame:
    return df[df["timestamp"] >= period_start].reset_index(drop=True)


def phase4_persistence(symbol: str, full_df: pd.DataFrame, reference: str,
                        account_id: str, token: str) -> pd.DataFrame:
    print(f"\n[FASE 4] Persistenza direzionale {symbol} vs {reference}...")
    ref_df = fetch_symbol_full_range(reference, account_id, token)

    rows = []
    for sym, df in [(symbol, full_df), (reference, ref_df)]:
        for period in PERIODS:
            window, period_start = slice_period(df, period)
            if window.empty:
                continue
            window = trim_warmup(window, period_start)
            if window.empty:
                continue
            r = compute_persistence(window)
            r["symbol"] = sym
            r["period"] = period
            rows.append(r)
            print(f"  [{sym}][{period}] persistenza: {r['persistenza_combinata']*100:.1f}%"
                  if pd.notna(r['persistenza_combinata']) else f"  [{sym}][{period}] n/a")

    return pd.DataFrame(rows)


def fetch_symbol_full_range(symbol: str, account_id: str, token: str) -> pd.DataFrame:
    rows, offset = [], 0
    while True:
        sql = (f"SELECT timestamp, open, high, low, close FROM ohlc_prices "
               f"WHERE symbol='{symbol}' ORDER BY timestamp LIMIT {CHUNK_SIZE_READ} OFFSET {offset}")
        batch = d1_query(sql, account_id, token)["result"][0]["results"]
        if not batch:
            break
        rows.extend(batch)
        offset += CHUNK_SIZE_READ
        if len(batch) < CHUNK_SIZE_READ:
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
# FASE 5 — Validazione statistica segnale (z-score vs random)
# ══════════════════════════════════════════════════════════════════

def make_random_signal_df(signal_df, n_long, n_short, rng):
    df = signal_df.copy()
    df["signal"] = None
    eligible = df.index[df["atr"].notna() & df["adx"].notna()]
    eligible = eligible[eligible < len(df) - 1]
    n_needed = n_long + n_short
    if len(eligible) < n_needed:
        n_needed = len(eligible)
        n_long = min(n_long, n_needed)
        n_short = n_needed - n_long
    chosen = rng.choice(eligible, size=n_needed, replace=False)
    rng.shuffle(chosen)
    df.loc[chosen[:n_long], "signal"] = "long"
    df.loc[chosen[n_long:n_long + n_short], "signal"] = "short"
    return df


def count_real_signals(signal_df):
    return (int((signal_df["signal"] == "long").sum()),
            int((signal_df["signal"] == "short").sum()))


def zscore(real_pnl, random_pnls):
    arr = np.array(random_pnls)
    std = arr.std(ddof=1)
    return 0.0 if std == 0 else (real_pnl - arr.mean()) / std


def run_single(symbol, inst, sig_df, capital0):
    engine_ = eng.BacktestEngine(capital0=capital0, instruments={symbol: inst})
    trades_df, metrics_df = engine_.run({symbol: sig_df})
    pnl = float(metrics_df["pnl_total"].iloc[0])
    n = int(metrics_df["num_trades"].iloc[0])
    dd_raw = metrics_df["max_drawdown_pct"].iloc[0]
    dd = float(dd_raw) if pd.notna(dd_raw) else 0.0
    return pnl, n, dd


def phase5_validation(symbol: str, inst: eng.InstrumentConfig, full_df: pd.DataFrame,
                       capital0: float, n_seeds: int = 30) -> pd.DataFrame:
    print(f"\n[FASE 5] Validazione statistica segnale {symbol} (z-score vs {n_seeds} random)...")
    rows = []
    for period in PERIODS:
        window, period_start = slice_period(full_df, period)
        sig = eng.generate_signals(window, inst)
        sig = trim_warmup(sig, period_start)
        real_pnl, n_trades, real_dd = run_single(symbol, inst, sig, capital0)

        if n_trades == 0:
            rows.append({"period": period, "num_trades": 0, "pnl_total": 0.0,
                         "max_drawdown_pct": 0.0, "z_score": np.nan})
            print(f"  [{period}] 0 trade.")
            continue

        n_long, n_short = count_real_signals(sig)
        random_pnls = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            rand_sig = make_random_signal_df(sig, n_long, n_short, rng)
            r_pnl, _, _ = run_single(symbol, inst, rand_sig, capital0)
            random_pnls.append(r_pnl)

        z = zscore(real_pnl, random_pnls)
        rows.append({"period": period, "num_trades": n_trades, "pnl_total": real_pnl,
                     "max_drawdown_pct": real_dd, "z_score": z})
        print(f"  [{period}] trades={n_trades} pnl={real_pnl:.1f} z={z:.2f}")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# FASE 6 — Calibrazione ATR × lookback
# ══════════════════════════════════════════════════════════════════

def phase6_calibration(symbol: str, base_inst: eng.InstrumentConfig, full_df: pd.DataFrame,
                        capital0: float, baseline_results: pd.DataFrame) -> dict:
    print(f"\n[FASE 6] Calibrazione ATR×lookback {symbol} (train {TRAIN_PERIOD})...")
    atr_grid = [1.0, 1.5, 2.0, 2.5, 3.0]
    lookback_grid = [15, 20, 30, 40, 50]

    def make_cfg(atr_mult, lookback):
        return dataclasses.replace(base_inst, atr_multiplier=atr_mult, breakout_lookback=lookback)

    def run_period(inst, period):
        window, period_start = slice_period(full_df, period)
        sig = eng.generate_signals(window, inst)
        sig = trim_warmup(sig, period_start)
        pnl, n, dd = run_single(symbol, inst, sig, capital0)
        ratio = pnl / abs(dd) if dd != 0 else 0.0
        return {"pnl_total": pnl, "num_trades": n, "max_drawdown_pct": dd, "pnl_dd_ratio": ratio}

    grid_rows = []
    for atr_mult in atr_grid:
        for lookback in lookback_grid:
            r = run_period(make_cfg(atr_mult, lookback), TRAIN_PERIOD)
            r["atr_mult"], r["lookback"] = atr_mult, lookback
            grid_rows.append(r)

    grid_df = pd.DataFrame(grid_rows)
    best = grid_df.loc[grid_df["pnl_dd_ratio"].idxmax()]
    best_atr, best_lb = best["atr_mult"], int(best["lookback"])
    print(f"  Migliore su train: ATR={best_atr} LB={best_lb} (ratio={best['pnl_dd_ratio']:.1f})")

    best_inst = make_cfg(best_atr, best_lb)
    verify_rows = []
    for period in VERIFY_PERIODS:
        r = run_period(best_inst, period)
        r["period"] = period
        verify_rows.append(r)
        print(f"  [verify {period}] pnl={r['pnl_total']:.1f} ratio={r['pnl_dd_ratio']:.1f}")
    verify_df = pd.DataFrame(verify_rows)

    calibrated_sum_pnl = best["pnl_total"] + verify_df["pnl_total"].sum()
    calibrated_worst_dd = min(best["max_drawdown_pct"], verify_df["max_drawdown_pct"].min())
    calibrated_ratio = calibrated_sum_pnl / abs(calibrated_worst_dd) if calibrated_worst_dd != 0 else 0.0

    baseline_valid = baseline_results.dropna(subset=["pnl_total"])
    baseline_sum_pnl = baseline_valid["pnl_total"].sum()
    baseline_worst_dd = baseline_valid["max_drawdown_pct"].min()
    baseline_ratio = baseline_sum_pnl / abs(baseline_worst_dd) if baseline_worst_dd != 0 else 0.0

    n_positive_verify = int((verify_df["pnl_total"] > 0).sum())
    margin = (calibrated_ratio / baseline_ratio - 1.0) if baseline_ratio != 0 else float("inf")
    promoted = (n_positive_verify >= 3) and (margin >= 0.10)

    print(f"\n  Calibrata: somma pnl={calibrated_sum_pnl:.1f} ratio={calibrated_ratio:.1f}")
    print(f"  Baseline (grezza): somma pnl={baseline_sum_pnl:.1f} ratio={baseline_ratio:.1f}")
    print(f"  Margine: {margin*100:+.1f}% | periodi verifica positivi: {n_positive_verify}/4")
    print(f"  {'CALIBRAZIONE PROMOSSA' if promoted else 'NON promossa — resta baseline grezza'}")

    grid_df.to_csv(f"results/{symbol.lower()}_pipeline_grid.csv", index=False)
    return {
        "best_atr": best_atr, "best_lookback": best_lb,
        "calibrated_ratio": calibrated_ratio, "baseline_ratio": baseline_ratio,
        "margin_vs_baseline": margin, "n_positive_verify": n_positive_verify,
        "promoted": promoted,
    }


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol")
    parser.add_argument("--reference", default="FTSE100")
    parser.add_argument("--capital", type=float, default=2000.0)
    args = parser.parse_args()

    symbol = args.symbol.strip().upper()
    reference = args.reference.strip().upper()
    capital0 = args.capital

    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        print("ERRORE: secrets CLOUDFLARE_API_TOKEN/CLOUDFLARE_ACCOUNT_ID mancanti.")
        sys.exit(1)

    os.makedirs("results", exist_ok=True)
    print(f"{'='*70}\nPIPELINE VALIDAZIONE INDICE: {symbol} (riferimento: {reference})\n{'='*70}")

    if symbol not in INSTRUMENT_REGISTRY:
        print(f"ERRORE: {symbol} non in INSTRUMENT_REGISTRY. Servono parametri IG "
              f"verificati (size minima, margine, point_value, spread da screenshot "
              f"'Get Info') prima di poter procedere. Aggiungili manualmente allo "
              f"script, poi rilancia.")
        sys.exit(1)
    base_inst = INSTRUMENT_REGISTRY[symbol]

    dukascopy_const = phase0_resolve_dukascopy_constant(symbol)
    full_df = phase1_download_ohlc(symbol, dukascopy_const)
    phase2_load_d1(symbol, full_df, account_id, token)
    corr_results = phase3_correlation(symbol, full_df, account_id, token)
    persistence_df = phase4_persistence(symbol, full_df, reference, account_id, token)
    persistence_df.to_csv(f"results/{symbol.lower()}_pipeline_persistence.csv", index=False)
    validation_df = phase5_validation(symbol, base_inst, full_df, capital0)
    validation_df.to_csv(f"results/{symbol.lower()}_pipeline_validation.csv", index=False)
    calibration_result = phase6_calibration(symbol, base_inst, full_df, capital0, validation_df)

    print(f"\n{'='*70}\nREPORT FINALE — {symbol}\n{'='*70}")
    print(f"Correlazione: {corr_results}")
    valid_persist = persistence_df[persistence_df['symbol'] == symbol]['persistenza_combinata'].mean()
    ref_persist = persistence_df[persistence_df['symbol'] == reference]['persistenza_combinata'].mean()
    print(f"Persistenza media {symbol}: {valid_persist*100:.1f}% | {reference}: {ref_persist*100:.1f}%")
    valid_z = validation_df.dropna(subset=["z_score"])
    print(f"Segnale grezzo: {int((valid_z['z_score']>0).sum())}/{len(valid_z)} periodi z-score positivo, "
          f"somma z={valid_z['z_score'].sum():.2f}, PnL positivo in "
          f"{int((validation_df['pnl_total']>0).sum())}/{len(validation_df)} periodi")
    print(f"Calibrazione: {calibration_result}")
    print(f"\nNessuna decisione automatica presa — valuta i numeri sopra prima di "
          f"procedere al confronto di coppia (asset_pair_comparison.py).")

    summary = pd.DataFrame([{
        "symbol": symbol, "reference": reference,
        **corr_results,
        "persistenza_media": valid_persist, "persistenza_reference": ref_persist,
        "n_periodi_z_positivo": int((valid_z['z_score']>0).sum()),
        "somma_z": valid_z['z_score'].sum(),
        "n_periodi_pnl_positivo": int((validation_df['pnl_total']>0).sum()),
        **calibration_result,
    }])
    summary.to_csv(f"results/{symbol.lower()}_pipeline_summary.csv", index=False)
    print(f"\nCompletato. Report: results/{symbol.lower()}_pipeline_summary.csv")


if __name__ == "__main__":
    main()
