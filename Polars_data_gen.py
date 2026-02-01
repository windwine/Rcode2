#%%
import polars as pl
import pyarrow as pa
import pandas as pd
import gc

#%%
pathname = "E:/laosongdata"

# ---- Load Parquet file ----
alldataEODs = (
    pl.read_parquet(f"{pathname}/alldataEODs.parquet")
    .rename({"close": "C"})
)
#%%

# ---- Construct alldataXs ----

alldataXs = (
    alldataEODs
    # .with_columns(
    # pl.col("Date").cast(pl.Date)
    # )
    # Ensure Date is actually a date type if it wasn't already:
    # .with_columns(
    #     pl.col("Date").str.strptime(pl.Date, strict=False)  # cast to Date
    # )
    .select(["C", "ID", "Date"])
    .filter(pl.col("C") > 0.01)
    .filter(pl.col("Date") >= pl.date(1999, 12, 1))
    .sort(["ID", "Date"])
    .with_columns([
        pl.col("Date").cum_count().over("ID").alias("age"),
    ])

)

alldataXs.head()

#%%
import time
t = time.time()
print()
spec = {
    "mu":   (pl.Expr.rolling_mean, [5, 10, 20, 60, 120, 250]),
    "sd":   (pl.Expr.rolling_std,  [5, 10, 20, 60, 120, 250]),
    "skew": (pl.Expr.rolling_skew, [5, 10, 20, 60, 120, 250]),
    "kurt": (pl.Expr.rolling_kurtosis, [5, 10, 20, 60, 120, 250]),
}

rolling_exprs = []

for name, (fn, wins) in spec.items():
    for w in wins:
        rolling_exprs.append(
            fn(pl.col("ret"), window_size=w)
              .over("ID")
              .alias(f"ret_{name}{w}")
        )

cumret_exprs = []

cumret_windows = range(1, 21)   # n = 1..20

for n in cumret_windows:
    cumret_exprs.append(
        (
            pl.col("C") / pl.col("C").shift(n) - 1
        )
        .over("ID")
        .alias(f"cumret_{n}")
    )

        
lf2 = (
    alldataXs.lazy()
    .sort(["ID", "Date"])
    .with_columns([
        (pl.int_range(0, pl.len()).over("ID") + 1).alias("hist"),
        (pl.col("C") / pl.col("C").shift(1) - 1).over("ID").alias("ret"),
    ])
    .with_columns(rolling_exprs)
    .with_columns(cumret_exprs)     # NEW cumulative returns
)

df = lf2.collect()
print(df.tail(10))
print(df.shape)
print(time.time() - t)
# %%
t = time.time()


# alldataXs: pandas DataFrame with at least columns ["ID", "Date", "C"]
# Ensure Date is datetime if needed:
# alldataXs["Date"] = pd.to_datetime(alldataXs["Date"])

df = alldataXs.to_pandas().sort_values(["ID", "Date"]).copy()

# --- group object ---
g = df.groupby("ID", sort=False)

# 1) hist: 1-based row number within each ID
df["hist"] = g.cumcount() + 1

# 2) ret: simple return within each ID
df["ret"] = g["C"].pct_change()  # same as C / C.shift(1) - 1

# 3) rolling stats spec (pandas uses strings for agg names)
windows = [5, 10, 20, 60, 120, 250]

# rolling mean/std
for w in windows:
    df[f"ret_mu{w}"] = g["ret"].transform(lambda s: s.rolling(w).mean())
    df[f"ret_sd{w}"] = g["ret"].transform(lambda s: s.rolling(w).std())

# rolling skew/kurt (kurtosis in pandas is Fisher by default, like many stats packages)
for w in windows:
    df[f"ret_skew{w}"] = g["ret"].transform(lambda s: s.rolling(w).skew())
    df[f"ret_kurt{w}"] = g["ret"].transform(lambda s: s.rolling(w).kurt())

# 4) cumret_1..20: C / C.shift(n) - 1 within each ID
for n in range(1, 21):
    df[f"cumret_{n}"] = g["C"].transform(lambda s: s / s.shift(n) - 1)

# df is your final equivalent of lf2.collect()
print(df.tail(10))
print(df.shape)
print(time.time() - t)


# %%
