import polars as pl, numpy as np, math, time
from stocksense.core.config import load_settings
from stocksense.features.build import load_features
from stocksense.labels.forward import load_labels
s = load_settings()
t0 = time.time()
F = load_features(s, "research").collect()
L = load_labels(s, "research").collect()
print("features", F.shape, F["date"].min(), F["date"].max(), " labels", L.shape, round(time.time()-t0,1), "s")
targets = ["r_y_oc", "r_y_c2o_5", "r_y_c2o_20"]
df = F.join(L.select(["date","sid"]+targets), on=["date","sid"], how="inner")
print("joined", df.shape)
feats = [c for c in F.columns if c not in ("date","sid") and not c.endswith("__raw")]
xs = [c for c in feats if df[c].n_unique() > 1000]
rows=[]
for f in xs:
    r=[f]
    for t in targets:
        d = df.select("date", f, t).drop_nulls()
        ic = d.group_by("date").agg(pl.len().alias("n"), pl.corr(f, t, method="spearman").alias("ic")).filter(pl.col("n")>=100).drop_nulls("ic")
        m=ic["ic"].mean(); sd=ic["ic"].std(); n=ic.height
        r += [m*100 if m is not None else float("nan"), (m/sd*math.sqrt(n)) if sd else float("nan")]
    rows.append(r)
cols=["feature"]+[f"{t}_{k}" for t in targets for k in ("ic","t")]
out=pl.DataFrame(rows, schema=cols, orient="row")
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(10); pl.Config.set_fmt_str_lengths(30)
print(out.with_columns(pl.col(c).round(2) for c in cols[1:]).sort("r_y_oc_t", descending=True, nulls_last=True))
