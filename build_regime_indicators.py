"""
build_regime_indicators.py — Infrastruttura per il nuovo filone di
ricerca "identificazione del regime di mercato" (22/07/2026). Calcola,
per OGNI barra da 30min (non solo ai momenti di segnale, a differenza
di corr_7d_entry), un set di indicatori di regime consolidati in
letteratura accademica:

- Variance Ratio (Lo-MacKinlay 1988, Review of Financial Studies):
  VR(k) = Var(ritorno k-periodi) / (k * Var(ritorno 1-periodo)).
  VR≈1 -> random walk, VR>1 -> trending (autocorrelazione positiva),
  VR<1 -> mean-reverting (autocorrelazione negativa). k=4 (scelta
  comune in letteratura). Calcolato su rolling window temporale.
- Autocorrelazione ritorni lag-1: versione piu' semplice, imparentata
  concettualmente con VR, calcolata come cross-check.
- Correlazione rolling DAX-FTSE100 (stessa metodologia di ieri,
  ricalcolata qui per essere nella stessa tabella).

Tutto calcolato su 2 finestre (7 e 21 giorni, stessa scala gia' usata
ieri per comparabilita') PER OGNI BARRA — non solo ai candidati — per
poter verificare la PERSISTENZA del regime nel tempo (quanto durano i
run dello stesso regime), che e' il primo test di affidabilita'
richiesto prima di collegare qualunque cosa al modello di trading.

Nuova tabella: market_regime_indicators (instrument, timestamp, close,
atr_pct, var_ratio_7d, var_ratio_21d, autocorr_7d, autocorr_21d,
corr_dax_ftse_7d, corr_dax_ftse_21d). Nessuna modifica a research_v6_
candidates o a engine.py.
"""
import os
import numpy as np
import pandas as pd
import requests

from ohlc_data_source import get_ohlc

CF_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
CF_API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]
D1_ID = "b9fbd4d6-7837-4d86-9c0f-ca60c0cf69e3"

VR_K = 4  # periodi per il variance ratio, scelta comune in letteratura


def d1(sql):
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{D1_ID}/query"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    resp = requests.post(url, json={"sql": sql}, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(data.get("errors"))
    return data["result"][0]


def rolling_variance_ratio(returns: pd.Series, k: int, window: str) -> pd.Series:
    """VR(k) = Var(ritorno k-periodi) / (k * Var(ritorno 1-periodo)),
    calcolato su finestra rolling temporale."""
    k_period_returns = returns.rolling(k).sum()  # somma di k log-return consecutivi
    var_k = k_period_returns.rolling(window).var()
    var_1 = returns.rolling(window).var()
    return var_k / (k * var_1)


def main():
    print("Creo tabella market_regime_indicators (se non esiste)...")
    d1("""CREATE TABLE IF NOT EXISTS market_regime_indicators (
        instrument TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        close REAL,
        atr_pct REAL,
        var_ratio_7d REAL,
        var_ratio_21d REAL,
        autocorr_7d REAL,
        autocorr_21d REAL,
        corr_dax_ftse_7d REAL,
        corr_dax_ftse_21d REAL,
        PRIMARY KEY (instrument, timestamp)
    )""")

    print("Scarico OHLC continuo DAX+FTSE100...")
    hist = {name: get_ohlc(name, CF_ACCOUNT_ID, CF_API_TOKEN) for name in ("DAX", "FTSE100")}

    print("Calcolo ATR (stessa logica di engine.py, Wilder 14) per ATR%...")
    import engine as eng
    atr_series = {}
    for name, df in hist.items():
        inst = eng.INSTRUMENTS[name]
        sig = eng.generate_signals(df, inst)
        atr_series[name] = sig.set_index("timestamp")["atr"]

    print("Calcolo log-return, variance ratio, autocorrelazione per strumento...")
    returns = {}
    log_returns = {}
    for name, df in hist.items():
        s = df.set_index("timestamp")["close"]
        log_returns[name] = np.log(s / s.shift(1))
        returns[name] = s.pct_change()

    indicators = {}
    for name in hist:
        lr = log_returns[name].dropna()
        vr7 = rolling_variance_ratio(lr, VR_K, "7D")
        vr21 = rolling_variance_ratio(lr, VR_K, "21D")
        ac7 = lr.rolling("7D").apply(lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 3 else np.nan, raw=False)
        ac21 = lr.rolling("21D").apply(lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 3 else np.nan, raw=False)
        indicators[name] = pd.DataFrame({
            "var_ratio_7d": vr7, "var_ratio_21d": vr21,
            "autocorr_7d": ac7, "autocorr_21d": ac21,
        })
        print(f"  {name}: indicatori calcolati")

    print("Calcolo correlazione rolling DAX-FTSE100 (7gg, 21gg)...")
    aligned = pd.concat([returns["DAX"].rename("dax"), returns["FTSE100"].rename("ftse")],
                         axis=1, sort=True).dropna()
    corr7 = aligned["dax"].rolling("7D").corr(aligned["ftse"])
    corr21 = aligned["dax"].rolling("21D").corr(aligned["ftse"])

    print("\nAssemblo tabella finale per strumento e scrivo su D1...")
    total_written = 0
    for name, df in hist.items():
        base = df.set_index("timestamp")[["close"]].copy()
        base["atr_pct"] = (atr_series[name] / base["close"]) * 100
        base = base.join(indicators[name])
        base["corr_dax_ftse_7d"] = corr7.reindex(base.index)
        base["corr_dax_ftse_21d"] = corr21.reindex(base.index)
        base = base.dropna(subset=["var_ratio_7d", "autocorr_7d"])
        base = base.reset_index()

        print(f"  {name}: {len(base)} righe da scrivere...")
        rows = base.to_dict("records")
        batch_size = 300
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            values = []
            for r in batch:
                def fv(v):
                    if v is None or (isinstance(v, float) and (pd.isna(v) or np.isinf(v))):
                        return "NULL"
                    return str(v)
                values.append(
                    f"('{name}', '{r['timestamp'].isoformat()}', {fv(r['close'])}, {fv(r['atr_pct'])}, "
                    f"{fv(r['var_ratio_7d'])}, {fv(r['var_ratio_21d'])}, "
                    f"{fv(r['autocorr_7d'])}, {fv(r['autocorr_21d'])}, "
                    f"{fv(r['corr_dax_ftse_7d'])}, {fv(r['corr_dax_ftse_21d'])})"
                )
            sql = ("INSERT OR REPLACE INTO market_regime_indicators "
                   "(instrument, timestamp, close, atr_pct, var_ratio_7d, var_ratio_21d, "
                   "autocorr_7d, autocorr_21d, corr_dax_ftse_7d, corr_dax_ftse_21d) VALUES "
                   + ",".join(values))
            d1(sql)
            total_written += len(batch)
            if total_written % 3000 == 0:
                print(f"    {total_written} righe scritte finora...")

    print(f"\nCompletato: {total_written} righe totali scritte in market_regime_indicators.")


if __name__ == "__main__":
    main()
