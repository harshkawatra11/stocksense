"""151 Trading Strategies sec 18.2: the quantile ANN on 15-minute bars (eqs. 521-537), walk-forward by month.

PLAN-M M5.4 d. Inputs at the close of bar b use bars <= b only: the normalised return R_hat(b), the EMA and EMSD
of R_hat at tau in {2, 4, 12, 24} bars (lambda = (tau-1)/(tau+1)) and the RSI at {12, 24, 48} bars. The label is
the quantile class (K = 5, edges from the training sample) of R_hat(b+1). The paper's t indexes time backwards;
its "R_hat(t)" input would be the label itself, so the latest completed bar b is used (docs/DECISIONS.md M5.4d).
Rule (eq. 537): buy at the open of bar b+1 when the predicted class is the top one; sell at its close.
"""

# ruff: noqa: E501
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from stocksense.core.config import Settings
from stocksense.costs.india import round_trip_cost_frac
from stocksense.data import io
from stocksense.data.ingest_intraday import load_bars
from stocksense.strategies.base import StrategySpec, register
from stocksense.strategies.k151 import roll_mean, roll_std, sh2

T1 = 100
TAUS = (2, 4, 12, 24)
RSI_TAUS = (12, 24, 48)
K_CLASSES = 5
HIDDEN = 32
SLOTS = 23  # entry bars 09:30..15:00 (bar starts), the last one closes by 15:15
LAST_ENTRY_MIN = 15 * 60  # an entry bar may start at 15:00 at the latest
N_FEATURES = 1 + 2 * len(TAUS) + len(RSI_TAUS)


def resample_15m(df5: pl.DataFrame) -> pl.DataFrame:
    """5-minute bars (symbol, ts, open, high, low, close, volume) to 15-minute bars aligned to 09:15; a
    bucket without all three 5-minute bars is dropped."""
    g = (
        df5.sort(["symbol", "ts"])
        .with_columns(pl.col("ts").dt.truncate("15m").alias("b"))
        .group_by(["symbol", "b"], maintain_order=True)
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") == 3)
        .drop("n")
        .rename({"b": "ts"})
    )
    return g.sort(["symbol", "ts"])


def _windows(x: np.ndarray, n: int) -> np.ndarray:
    """(T, n) matrix whose row b holds x[b], x[b-1], ..., x[b-n+1] (NaN before the start)."""
    pad = np.concatenate([np.full(n - 1, np.nan), x])
    return sliding_window_view(pad, n)[:, ::-1]


def ema_emsd(rhat: np.ndarray, tau: int) -> tuple[np.ndarray, np.ndarray]:
    """Eqs. 526-527 over the last tau bars b, b-1, ... with weights lambda^k, lambda = (tau-1)/(tau+1)."""
    lam = (tau - 1.0) / (tau + 1.0)
    k = lam ** np.arange(tau)
    win = _windows(rhat, tau)
    with np.errstate(invalid="ignore", divide="ignore"):
        ema = (1 - lam) / (1 - lam**tau) * (win * k).sum(axis=1)
        var = (1 - lam) / (lam - lam**tau) * (k * (win - ema[:, None]) ** 2).sum(axis=1)
    return ema, np.sqrt(np.maximum(var, 0.0))


def rsi(rhat: np.ndarray, tau: int) -> np.ndarray:
    """Eqs. 528-529: sum of positive over sum of absolute R_hat in the last tau bars."""
    win = _windows(rhat, tau)
    up = np.maximum(win, 0).sum(axis=1)
    dn = np.maximum(-win, 0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(win).all(axis=1), up / (up + dn), np.nan)


def normalised_return(close: np.ndarray, t1: int = T1) -> np.ndarray:
    """Eqs. 521-525: (R(b) - mean of the previous t1 returns)/std of the previous t1 returns."""
    r = close / sh2(close, 1) - 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return (r - sh2(roll_mean(r, t1), 1)) / sh2(roll_std(r, t1), 1)


def ann_inputs(close: np.ndarray, t1: int = T1) -> tuple[np.ndarray, np.ndarray]:
    """(X [T, 12], R_hat [T]) of one symbol's concatenated 15-minute closes; X[b] uses bars <= b only."""
    rhat = normalised_return(np.asarray(close, dtype=float), t1)
    cols = [rhat]
    for tau in TAUS:
        e, s = ema_emsd(rhat, tau)
        cols += [e, s]
    cols += [rsi(rhat, tau) for tau in RSI_TAUS]
    return np.column_stack(cols), rhat


def quantile_classes(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Eq. 530: class index 0..K-1 of each value given the K-1 ascending edges."""
    return np.searchsorted(edges, x, side="right")


def _bar_table(bars: pl.DataFrame) -> dict[str, np.ndarray]:
    """Stack every symbol's inputs, labels and trade prices into flat arrays (one row per bar b)."""
    xs, ys, ent_o, ent_c, month, day, sym, tsl, ok_rows = [], [], [], [], [], [], [], [], []
    for name, g in bars.group_by("symbol", maintain_order=True):
        g = g.sort("ts")
        c = g["close"].to_numpy().astype(float)
        if len(c) < T1 + 60:
            continue
        x, rhat = ann_inputs(c)
        d = g["ts"].dt.date().to_numpy().astype("datetime64[D]")
        mod = (g["ts"].dt.hour().cast(pl.Int64) * 60 + g["ts"].dt.minute().cast(pl.Int64)).to_numpy()
        nxt_same = np.r_[d[1:] == d[:-1], False]
        y = np.r_[rhat[1:], np.nan]
        o_n = np.r_[g["open"].to_numpy()[1:], np.nan]
        c_n = np.r_[c[1:], np.nan]
        mod_n = np.r_[mod[1:], 10**6]
        ok = nxt_same & np.isfinite(x).all(axis=1) & np.isfinite(y) & (mod_n <= LAST_ENTRY_MIN)
        n = len(c)
        xs.append(x), ys.append(y), ent_o.append(o_n), ent_c.append(c_n)
        month.append(d.astype("datetime64[M]")), day.append(np.r_[d[1:], d[-1:]])
        sym.append(np.full(n, name[0], dtype=object)), tsl.append(g["ts"].to_numpy())
        ok_rows.append(ok)
    if not xs:
        return {}
    return {
        "x": np.vstack(xs), "y": np.concatenate(ys), "o": np.concatenate(ent_o), "c": np.concatenate(ent_c),
        "month": np.concatenate(month), "day": np.concatenate(day), "sym": np.concatenate(sym),
        "ts": np.concatenate(tsl), "ok": np.concatenate(ok_rows),
    }  # fmt: skip


def train_predict(
    xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, seed: int = 1, epochs: int = 3, batch: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the 2x32 ReLU softmax network (eqs. 532-536, Adam, cross-entropy) and return (class probabilities of
    xte [N, K], the class edges used). Quantile edges and input scaling come from the training sample only."""
    import torch

    edges = np.quantile(ytr, np.linspace(0, 1, K_CLASSES + 1)[1:-1])
    cls = quantile_classes(ytr, edges)
    mu, sd = xtr.mean(axis=0), xtr.std(axis=0) + 1e-9
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, K_CLASSES),
    )  # fmt: skip
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt = torch.tensor((xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(cls, dtype=torch.long)
    for _ in range(epochs):
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(xt), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(xt[idx]), yt[idx]).backward()
            opt.step()
    with torch.no_grad():
        p = torch.softmax(net(torch.tensor((xte - mu) / sd, dtype=torch.float32)), dim=1).numpy()
    return p, edges


def walk_forward(bars: pl.DataFrame, min_train_months: int = 3, seed: int = 1) -> pl.DataFrame:
    """Per month, train on every earlier month and flag the test month's top-class bars.

    Returns the trades (symbol, ts of the signal bar, date, ret) where ret = close/open - 1 of the NEXT 15-minute
    bar (bought at its open, sold at its close), before costs.
    """
    t = _bar_table(bars)
    schema = {"symbol": pl.Utf8, "ts": bars.schema["ts"], "date": pl.Date, "ret": pl.Float64}
    if not t:
        return pl.DataFrame(schema=schema)
    months = np.unique(t["month"][t["ok"]])
    rows = []
    for m in months[min_train_months:]:
        tr = t["ok"] & (t["month"] < m)
        te = t["ok"] & (t["month"] == m)
        if tr.sum() < 1000 or not te.any():
            continue
        p, _ = train_predict(t["x"][tr], t["y"][tr], t["x"][te], seed=seed)
        hit = p.argmax(axis=1) == K_CLASSES - 1
        idx = np.flatnonzero(te)[hit]
        rows.append(
            pl.DataFrame(
                {
                    "symbol": t["sym"][idx].tolist(),
                    "ts": t["ts"][idx],
                    "date": t["day"][idx],
                    "ret": t["c"][idx] / t["o"][idx] - 1.0,
                },
                schema=schema,
            )
        )
    return pl.concat(rows) if rows else pl.DataFrame(schema=schema)


def daily_from_trades(trades: pl.DataFrame, settings: Settings, capital_inr: float) -> pl.DataFrame:
    """(date, gross, ret_gross, net, net_2x): each of the 23 entry slots of a day holds 1/23 of the capital, split
    equally over that bar's signals; `gross` is the share of slots used; costs are the intraday round trip."""
    if not trades.height:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "gross": pl.Float64,
                "ret_gross": pl.Float64,
                "net": pl.Float64,
                "net_2x": pl.Float64,
            }
        )
    per_bar = trades.group_by(["date", "ts"]).agg(pl.col("ret").mean().alias("r"), pl.len().alias("n"))
    rows = []
    for d, g in per_bar.group_by("date", maintain_order=True):
        cost = [
            round_trip_cost_frac("intraday", capital_inr / SLOTS / max(int(n), 1), settings.costs)
            for n in g["n"]
        ]
        gross_r = float(g["r"].sum()) / SLOTS
        c = float(np.sum(cost)) / SLOTS
        rows.append((d[0], g.height / SLOTS, gross_r, gross_r - c, gross_r - 2 * c))
    return pl.DataFrame(rows, schema=["date", "gross", "ret_gross", "net", "net_2x"], orient="row").sort(
        "date"
    )


def u200_symbols(settings: Settings, end: date) -> list[str]:
    """Symbols of the last stored u200 membership on or before `end`."""
    uni = io.read(settings, "panel_universe", start=None, end=end, columns=["date", "sid", "u200"]).collect()
    last = uni["date"].max()
    sids = uni.filter((pl.col("date") == last) & pl.col("u200"))["sid"].to_list()
    px = io.read(settings, "panel_daily", start=last, end=last, columns=["sid", "symbol"]).collect()
    return sorted(set(px.filter(pl.col("sid").is_in(sids))["symbol"].to_list()))


def ann_run(settings: Settings, start: date, end: date, capital_inr: float | None = None) -> pl.DataFrame:
    """The walk-forward ANN strategy on stored 5-minute bars of the u200 (a thin wrapper for M5.7)."""
    syms = u200_symbols(settings, end)
    df5 = load_bars(settings, syms, start, end, interval="5m", basis="adjusted")
    if not df5.height:
        return daily_from_trades(pl.DataFrame(), settings, 1.0)
    trades = walk_forward(resample_15m(df5))
    cap = settings.gate.research.reference_capital_inr if capital_inr is None else capital_inr
    return daily_from_trades(trades, settings, cap)


register(
    StrategySpec(
        key="k151_ann_15m",
        name="151 ANN on 15-minute bars",
        source="k151",
        tier="none",
        family="m-intra",
        timeframe="5m",
        kind="event",
        direction="long",
        exits=("eod",),
        default_exit="eod",
        signal=None,
        timing=ann_run,  # type: ignore[arg-type]
        description="Quantile ANN (5 classes) on 15-minute bars of the u200, retrained monthly; buys the top class.",
        reference="151 Trading Strategies sec 18.2 (eqs. 521-537)",
    )
)
