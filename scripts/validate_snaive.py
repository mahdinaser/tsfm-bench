"""Validate the metric implementation: seasonal naive on FULL M4 vs published M4 table."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsfm_bench.data import _load_m4, GROUPS
from tsfm_bench.metrics import smape, mase_scale

# Published M4 sNaive row, "Evaluation and Ranks.xlsx" -> Point Forecasts-Frequency
PUB = {
 "Yearly":   dict(smape=16.34, mase=3.974, h=6,  m=1),
 "Quarterly":dict(smape=12.52, mase=1.602, h=8,  m=4),
 "Monthly":  dict(smape=15.99, mase=1.260, h=18, m=12),
 "Weekly":   dict(smape=9.161, mase=2.777, h=13, m=1),
 "Daily":    dict(smape=3.045, mase=3.278, h=14, m=1),
 "Hourly":   dict(smape=13.91, mase=1.193, h=48, m=24),
}

rows, all_sm, all_ms = [], [], []
for freq, p in PUB.items():
    ids, train, test = _load_m4(freq)
    h, m = p["h"], p["m"]
    sm_series, ms_series = [], []
    for tr, te in zip(train, test):
        fc = np.resize(tr[-m:], h) if m > 1 else np.repeat(tr[-1], h)
        sm_series.append(np.mean(smape(te, fc)))
        sc = mase_scale(tr, m)
        ms_series.append(np.mean(np.abs(te - fc)) / sc)
    sm, ms = float(np.mean(sm_series)), float(np.mean(ms_series))
    all_sm.extend(sm_series); all_ms.extend(ms_series)
    rows.append(dict(freq=freq, n=len(ids),
                     smape=round(sm,3), smape_pub=p["smape"], smape_d=round(sm-p["smape"],3),
                     mase=round(ms,3),  mase_pub=p["mase"],  mase_d=round(ms-p["mase"],3)))
rows.append(dict(freq="TOTAL", n=len(all_sm),
                 smape=round(float(np.mean(all_sm)),3), smape_pub=14.66,
                 smape_d=round(float(np.mean(all_sm))-14.66,3),
                 mase=round(float(np.mean(all_ms)),3), mase_pub=np.nan, mase_d=np.nan))
df = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(df.to_string(index=False))
ok = all(abs(r["smape_d"]) < 0.01 and (np.isnan(r["mase_d"]) or abs(r["mase_d"]) < 0.01) for r in rows)
print("\nACCEPTANCE:", "PASS — reproduces published M4 sNaive" if ok else "FAIL — investigate")
