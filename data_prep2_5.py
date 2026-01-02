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
    .rename({"C": "Price"})
)

# ---- Extract SPX ----
SPX = alldataXs.filter(pl.col("ID") == "SH000001")

# ---- Construct alldata (remove SPX) ----
alldata = (
    alldataXs
    .filter(pl.col("ID") != "SH000001")
    .select(["Date", "Price", "ID", "age"])
)

# ---- Benchmark as pandas time series ----
benchmark = pd.Series(
    SPX.get_column("Price").to_list(),
    index=pd.to_datetime(SPX.get_column("Date").to_list())
)

benchmark = SPX
# ---- Cleanup ----
# del alldataXs
# gc.collect()


#%%
# Flat polars + numpy version of rebalance loop
import polars as pl
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

lookback_months = 13
forward_months = 1
lookback = lookback_months * 4 + 1

# Split benchmark and alldata from alldataXs
df = alldataXs.to_pandas()

SPX = df[df["ID"] == "SH000001"].copy()
SPX["Date"] = pd.to_datetime(SPX["Date"])
benchmark = pd.Series(SPX["Price"].values, index=SPX["Date"], name="Price").sort_index()

alldata = df[df["ID"] != "SH000001"][["Date", "Price", "ID", "age"]].copy()
alldata["Date"] = pd.to_datetime(alldata["Date"])
dates = benchmark.index.to_list()


wkdays = ["Monday", "Tuesday", "Wednesday", "Thursday"]

for actdays in range(5, 6):
    if actdays <= 4:
        dayname = wkdays[actdays - 1]
        tuesdayindex = benchmark.index.weekday == actdays - 1
        actiondays = benchmark[tuesdayindex]
        actiondates = actiondays.index.to_list()
        if actiondates[-1] != dates[-1]: # we are appending the last date to the end of actiondates
            actiondays = pd.concat([actiondays, benchmark[-1:]])
            actiondates = actiondays.index.to_list()
    else:
        actiondays = benchmark.groupby(benchmark.index.to_period("W")).tail(1)
        actiondates = actiondays.index.to_list()

        
    # some check for actiondates
    # Convert actiondates to a pandas DatetimeIndex
    actiondates_index = pd.to_datetime(actiondates)

    # Get the weekdays (0=Monday, 6=Sunday)
    weekdays = actiondates_index.weekday

    # Print the weekdays
    # print(weekdays)
    
    # some data info, tempdata 1 year of data until reb_date(y). 
    # df_price is the weekly price data until the reb_date(y)(ID as index) 
    # df_daily is the daily price data until the reb_date(y)(ID as index)
    # df_old is the 21 daily price data until the prev_reb_date(y-1)(ID as index)
    # df_long is the 252 daily price data until the prev_reb_date(y-1)(ID as index)
    # temprets is the returns from df_long (252 days)
    
    results = []
    # for i in range(lookback + 5, len(actiondates)):
    for i in range(lookback + 5, lookback + 10):
        reb_date = actiondates[i]
        prev_reb_date = actiondates[i - 1]
        print(reb_date)

        temp_ids = alldata.loc[(alldata["Date"] == reb_date) & (alldata["age"] >= lookback * 5), "ID"].unique()
        tempdata = alldata[alldata["ID"].isin(temp_ids)].copy()

        daterange = actiondates[i - (lookback + forward_months - 1):i + 1][::-1] # reverse the order 1 is the most recent date
        allprices = [] # this is the weekly price data as you are looping through the actiondates
        for d in daterange:
            temp = tempdata[tempdata["Date"] == d][["ID", "Price"]].copy()
            temp = temp.rename(columns={"Price": f"x_{d.date()}"})
            allprices.append(temp)
        price_wide = allprices[0]
        for p in allprices[1:]:
            price_wide = price_wide.merge(p, on="ID", how="outer")
        df_price = price_wide.set_index("ID").T.fillna(method="ffill").fillna(method="bfill").T

        current_idx = dates.index(reb_date)
        ddates = dates[current_idx - 252 + 1: current_idx + 1][::-1]
        alld = [] # this is the daily price data as you are looping through the dates from the benchmark series
        for d in ddates:
            temp = tempdata[tempdata["Date"] == d][["ID", "Price"]].copy()
            temp = temp.rename(columns={"Price": f"d_{d.date()}"})
            alld.append(temp)
        daily_wide = alld[0]
        for s in alld[1:]:
            daily_wide = daily_wide.merge(s, on="ID", how="outer")
        df_daily = daily_wide.set_index("ID").T.fillna(method="ffill").fillna(method="bfill").T

        if df_price.shape[0] != df_daily.shape[0]:
            print("bad data not the same # of IDs", actiondates[i])
            continue

        temp_diff = df_price.iloc[:, 1] - df_daily.iloc[:, 1]
        if abs(temp_diff).sum() > 0.5:
            print("misaligned daily/monthly price on", actiondates[i])

        df_price.iloc[:, 1] = df_daily.iloc[:, 1]  # Align starting point

        # Historical daily window for model construction (t-1)
        current_idx = dates.index(prev_reb_date)
        ii = current_idx
        allpricesd_old = []
        for k in range(21):
            d = dates[ii - k]
            temp = tempdata[tempdata["Date"] == d][["ID", "Price"]].copy()
            temp = temp.rename(columns={"Price": f"xd_{k}"})
            allpricesd_old.append(temp)
        price_old_wide = allpricesd_old[0]
        for p in allpricesd_old[1:]:
            price_old_wide = price_old_wide.merge(p, on="ID", how="outer")
        df_old = price_old_wide.set_index("ID").T.fillna(method="ffill").fillna(method="bfill").T

        # New tempallpricesd for long window (252 days)
        tempallpricesd = []
        tempspxd = []
        for k in range(252):
            d = dates[ii - k]
            temp = tempdata[tempdata["Date"] == d][["ID", "Price"]].copy()
            temp = temp.rename(columns={"Price": f"xd_{k}"})
            tempallpricesd.append(temp)
            tempspxd.append(benchmark[d])
        df_long = tempallpricesd[0]
        for p in tempallpricesd[1:]:
            df_long = df_long.merge(p, on="ID", how="outer")

        df_long = df_long.set_index("ID").T.fillna(method="ffill").fillna(method="bfill").T
        df_long = df_long.iloc[:, ::-1]  # reverse to t, t+1...
        X = np.log(df_long.values)
        temprets = np.diff(X, axis=1)
        temprets[np.isnan(temprets)] = 0
        spxret = np.diff(np.log(np.array(tempspxd[::-1]))) # you need to reverse the order of spx as in r it was auto ranked by time

        ndays = temprets.shape[1]
        tempmu = temprets.mean(axis=1)
        tempsd = temprets.std(axis=1)
        temp_short_vol = temprets[:, -20:].std(axis=1)
        tempskew = skew(temprets, axis=1, nan_policy="omit")
        tempkurt = kurtosis(temprets, axis=1, nan_policy="omit")
        tempsharpe = tempmu / (tempsd + 1e-8)

        tempalpha = np.zeros_like(tempsharpe)
        tempbeta = np.zeros_like(tempsharpe)
        tempcorr = np.zeros_like(tempsharpe)
        tempMax = np.zeros_like(tempsharpe)

        for iii in range(len(temprets)):
            y = temprets[iii, :]
            mask = np.isfinite(y) & np.isfinite(spxret)
            if mask.sum() < 10:
                continue
            Xreg = np.vstack([np.ones(mask.sum()), spxret[mask]]).T
            b, _, _, _ = np.linalg.lstsq(Xreg, y[mask], rcond=None)
            tempalpha[iii] = b[0]
            tempbeta[iii] = b[1]
            tempcorr[iii] = np.corrcoef(y[mask], spxret[mask])[0, 1]
            tempMax[iii] = np.mean(np.sort(y[-20:])[::-1][:5])

        temprets2 = temprets[:, -21:]
        spxret2 = spxret[-21:]
        ndays2 = temprets2.shape[1]
        tempmu2 = temprets2.mean(axis=1)
        tempsd2 = temprets2.std(axis=1)
        temp_short_vol2 = temprets2[:, -5:].std(axis=1)
        tempskew2 = skew(temprets2, axis=1, nan_policy="omit")
        tempkurt2 = kurtosis(temprets2, axis=1, nan_policy="omit")
        tempsharpe2 = tempmu2 / (tempsd2 + 1e-8)

        tempalpha2 = np.zeros_like(tempsharpe2)
        tempbeta2 = np.zeros_like(tempsharpe2)
        tempcorr2 = np.zeros_like(tempsharpe2)
        tempMax2 = np.zeros_like(tempsharpe2)

        for iii in range(len(temprets2)):
            y = temprets2[iii, :]
            mask = np.isfinite(y) & np.isfinite(spxret2)
            if mask.sum() < 10:
                continue
            Xreg = np.vstack([np.ones(mask.sum()), spxret2[mask]]).T
            b, _, _, _ = np.linalg.lstsq(Xreg, y[mask], rcond=None)
            tempalpha2[iii] = b[0]
            tempbeta2[iii] = b[1]
            tempcorr2[iii] = np.corrcoef(y[mask], spxret2[mask])[0, 1]
            tempMax2[iii] = np.mean(np.sort(y[-5:])[::-1][:5])

        allpricedfactor_old = pd.DataFrame({
            "ID": df_long.index,  # Include the ID column
            "tempsd": tempsd,
            "tempskew": tempskew,
            "tempkurt": tempkurt,
            "tempsharpe": tempsharpe,
            "tempalpha": tempalpha,
            "tempbeta": tempbeta,
            "tempcorr": tempcorr,
            "temp_short_vol": temp_short_vol,
            "tempMax": tempMax,
            "tempsd2": tempsd2,
            "tempskew2": tempskew2,
            "tempkurt2": tempkurt2,
            "tempsharpe2": tempsharpe2,
            "tempalpha2": tempalpha2,
            "tempbeta2": tempbeta2,
            "tempcorr2": tempcorr2,
            "temp_short_vol2": temp_short_vol2,
            "tempMax2": tempMax2
        }).reset_index(drop=True)  # Reset the index to avoid duplications
        
        # === CONTINUATION: Generate rets ===
        
        # all the xs and y do not have index but with an ID column
        
        # the prices here are not used as X since it is based on to the new rebdate. It is redundant but it is the way the original code was written
        allprices = df_price.copy()
        allprices.columns = ["ID"] + [f"x{i}" for i in range(1, df_price.shape[1])]
        allpricesd = df_daily.copy()
        allpricesd.columns = ["ID"] + [f"xd{i}" for i in range(1, df_daily.shape[1])]
        # allpricesd_old.columns = ["ID"] + [f"xd_old{i}" for i in range(1, df_old.shape[1])]

        # cumulative rets
        df_old.reset_index(drop=False, inplace=True)
        allretsd_old = 1 / (df_old.iloc[:, 2:].div(df_old.iloc[:, 1], axis=0))
        allretsd_old["ID"] = df_old.iloc[:, 0].values
        
        df_daily.reset_index(drop=False, inplace=True)
        allretsd_new = 1 / (df_daily.iloc[:, 2:].div(df_daily.iloc[:, 1], axis=0))
        allretsd_new["ID"] = df_daily.iloc[:, 0].values

        df_price.reset_index(drop=False, inplace=True)
        allrets = 1 / (df_price.iloc[:, 2:].div(df_price.iloc[:, 1], axis=0))
        allrets["ID"] = df_price.iloc[:, 0].values

        allrets_old = 1 / (df_price.iloc[:, 3:].div(df_price.iloc[:, 2], axis=0))
        allrets_old["ID"] = df_price.iloc[:, 0].values

        # define X and y
        y = allrets.iloc[:, [0]].copy()
        y.columns = ["y"]
        y["ID"] = allrets["ID"].values
        y.reset_index(drop=True, inplace=True)

        x1 = allrets_old.iloc[:, :-1].copy()
        x1.columns = [f"m{j+1}" for j in range(x1.shape[1])]
        x1["ID"] = allrets_old["ID"].values
        x1.reset_index(drop=True, inplace=True)

        x2 = allretsd_old.iloc[:, :20].copy()
        x2.columns = [f"d{j+1}" for j in range(20)]
        x2["ID"] = allretsd_old["ID"].values
        x2.reset_index(drop=True, inplace=True)

        xs = pd.merge(x1, x2, on="ID")
        xs = pd.merge(xs, allpricedfactor_old, on="ID")

        # standardize y
        y["classy"] = (y["y"] - y["y"].mean()) / y["y"].std()

        currentdata = pd.merge(y, xs, on="ID")
        currentdata["Date"] = actiondates[i - 1]
        
        results.append(currentdata)

    results_df = pd.concat(results, ignore_index=True)


# %%
from joblib import Parallel, delayed

def f(i):
    # Your function logic here
    return i * i
results = Parallel(n_jobs=-1)(
    delayed(f)(i)
    for i in range(1, 101)
)

# %%
wkdays = ["Monday", "Tuesday", "Wednesday", "Thursday"]

for actdays in range(1, 6):
    if actdays <= 4:
        dayname = wkdays[actdays - 1]
        tuesdayindex = benchmark.index.weekday == actdays - 1
        actiondays = benchmark[tuesdayindex]
        actiondates = actiondays.index.to_list()
        if actiondates[-1] != dates[-1]: # we are appending the last date to the end of actiondates
            actiondays = pd.concat([actiondays, benchmark[-1:]])
            actiondates = actiondays.index.to_list()
    else:
        actiondays = benchmark.groupby(benchmark.index.to_period("W")).tail(1)
        actiondates = actiondays.index.to_list()

        
    # some check for actiondates
    # Convert actiondates to a pandas DatetimeIndex
    actiondates_index = pd.to_datetime(actiondates)

    # Get the weekdays (0=Monday, 6=Sunday)
    weekdays = actiondates_index.weekday

    # Print the weekdays
    # print(weekdays)
    
    # some data info, tempdata 1 year of data until reb_date(y). 
    # df_price is the weekly price data until the reb_date(y)(ID as index) 
    # df_daily is the daily price data until the reb_date(y)(ID as index)
    # df_old is the 21 daily price data until the prev_reb_date(y-1)(ID as index)
    # df_long is the 252 daily price data until the prev_reb_date(y-1)(ID as index)
    # temprets is the returns from df_long (252 days)
    def worker_func(i, actiondates, alldata, lookback):
        """Run one parallel iteration."""
        reb_date = actiondates[i]
        prev_reb_date = actiondates[i - 1]
        print(reb_date)

        temp_ids = alldata.loc[(alldata["Date"] == reb_date) & (alldata["age"] >= lookback * 5), "ID"].unique()
        tempdata = alldata[alldata["ID"].isin(temp_ids)].copy()

        daterange = actiondates[i - (lookback + forward_months - 1):i + 1][::-1] # reverse the order 1 is the most recent date
        allprices = [] # this is the weekly price data as you are looping through the actiondates
        for d in daterange:
            temp = tempdata[tempdata["Date"] == d][["ID", "Price"]].copy()
            temp = temp.rename(columns={"Price": f"x_{d.date()}"})
            allprices.append(temp)
        price_wide = allprices[0]
        for p in allprices[1:]:
            price_wide = price_wide.merge(p, on="ID", how="outer")
        df_price = price_wide.set_index("ID").T.fillna(method="ffill").fillna(method="bfill").T

        current_idx = dates.index(reb_date)
        ddates = dates[current_idx - 252 + 1: current_idx + 1][::-1]
        alld = [] # this is the daily price data as you are looping through the dates from the benchmark series
        for d in ddates:
            temp = tempdata[tempdata["Date"] == d][["ID", "Price"]].copy()
            temp = temp.rename(columns={"Price": f"d_{d.date()}"})
            alld.append(temp)
        daily_wide = alld[0]
        for s in alld[1:]:
            daily_wide = daily_wide.merge(s, on="ID", how="outer")
        df_daily = daily_wide.set_index("ID").T.fillna(method="ffill").fillna(method="bfill").T

        if df_price.shape[0] != df_daily.shape[0]:
            print("bad data not the same # of IDs", actiondates[i])
            return None

        temp_diff = df_price.iloc[:, 1] - df_daily.iloc[:, 1]
        if abs(temp_diff).sum() > 0.5:
            print("misaligned daily/monthly price on", actiondates[i])

        df_price.iloc[:, 1] = df_daily.iloc[:, 1]  # Align starting point

        # Historical daily window for model construction (t-1)
        current_idx = dates.index(prev_reb_date)
        ii = current_idx
        allpricesd_old = []
        for k in range(21):
            d = dates[ii - k]
            temp = tempdata[tempdata["Date"] == d][["ID", "Price"]].copy()
            temp = temp.rename(columns={"Price": f"xd_{k}"})
            allpricesd_old.append(temp)
        price_old_wide = allpricesd_old[0]
        for p in allpricesd_old[1:]:
            price_old_wide = price_old_wide.merge(p, on="ID", how="outer")
        df_old = price_old_wide.set_index("ID").T.fillna(method="ffill").fillna(method="bfill").T

        # New tempallpricesd for long window (252 days)
        tempallpricesd = []
        tempspxd = []
        for k in range(252):
            d = dates[ii - k]
            temp = tempdata[tempdata["Date"] == d][["ID", "Price"]].copy()
            temp = temp.rename(columns={"Price": f"xd_{k}"})
            tempallpricesd.append(temp)
            tempspxd.append(benchmark[d])
        df_long = tempallpricesd[0]
        for p in tempallpricesd[1:]:
            df_long = df_long.merge(p, on="ID", how="outer")

        df_long = df_long.set_index("ID").T.fillna(method="ffill").fillna(method="bfill").T
        df_long = df_long.iloc[:, ::-1]  # reverse to t, t+1...
        X = np.log(df_long.values)
        temprets = np.diff(X, axis=1)
        temprets[np.isnan(temprets)] = 0
        spxret = np.diff(np.log(np.array(tempspxd[::-1]))) # you need to reverse the order of spx as in r it was auto ranked by time

        ndays = temprets.shape[1]
        tempmu = temprets.mean(axis=1)
        tempsd = temprets.std(axis=1)
        temp_short_vol = temprets[:, -20:].std(axis=1)
        tempskew = skew(temprets, axis=1, nan_policy="omit")
        tempkurt = kurtosis(temprets, axis=1, nan_policy="omit")
        tempsharpe = tempmu / (tempsd + 1e-8)

        tempalpha = np.zeros_like(tempsharpe)
        tempbeta = np.zeros_like(tempsharpe)
        tempcorr = np.zeros_like(tempsharpe)
        tempMax = np.zeros_like(tempsharpe)

        for iii in range(len(temprets)):
            y = temprets[iii, :]
            mask = np.isfinite(y) & np.isfinite(spxret)
            if mask.sum() < 10:
                continue
            Xreg = np.vstack([np.ones(mask.sum()), spxret[mask]]).T
            b, _, _, _ = np.linalg.lstsq(Xreg, y[mask], rcond=None)
            tempalpha[iii] = b[0]
            tempbeta[iii] = b[1]
            tempcorr[iii] = np.corrcoef(y[mask], spxret[mask])[0, 1]
            tempMax[iii] = np.mean(np.sort(y[-20:])[::-1][:5])

        temprets2 = temprets[:, -21:]
        spxret2 = spxret[-21:]
        ndays2 = temprets2.shape[1]
        tempmu2 = temprets2.mean(axis=1)
        tempsd2 = temprets2.std(axis=1)
        temp_short_vol2 = temprets2[:, -5:].std(axis=1)
        tempskew2 = skew(temprets2, axis=1, nan_policy="omit")
        tempkurt2 = kurtosis(temprets2, axis=1, nan_policy="omit")
        tempsharpe2 = tempmu2 / (tempsd2 + 1e-8)

        tempalpha2 = np.zeros_like(tempsharpe2)
        tempbeta2 = np.zeros_like(tempsharpe2)
        tempcorr2 = np.zeros_like(tempsharpe2)
        tempMax2 = np.zeros_like(tempsharpe2)

        for iii in range(len(temprets2)):
            y = temprets2[iii, :]
            mask = np.isfinite(y) & np.isfinite(spxret2)
            if mask.sum() < 10:
                continue
            Xreg = np.vstack([np.ones(mask.sum()), spxret2[mask]]).T
            b, _, _, _ = np.linalg.lstsq(Xreg, y[mask], rcond=None)
            tempalpha2[iii] = b[0]
            tempbeta2[iii] = b[1]
            tempcorr2[iii] = np.corrcoef(y[mask], spxret2[mask])[0, 1]
            tempMax2[iii] = np.mean(np.sort(y[-5:])[::-1][:5])

        allpricedfactor_old = pd.DataFrame({
            "ID": df_long.index,  # Include the ID column
            "tempsd": tempsd,
            "tempskew": tempskew,
            "tempkurt": tempkurt,
            "tempsharpe": tempsharpe,
            "tempalpha": tempalpha,
            "tempbeta": tempbeta,
            "tempcorr": tempcorr,
            "temp_short_vol": temp_short_vol,
            "tempMax": tempMax,
            "tempsd2": tempsd2,
            "tempskew2": tempskew2,
            "tempkurt2": tempkurt2,
            "tempsharpe2": tempsharpe2,
            "tempalpha2": tempalpha2,
            "tempbeta2": tempbeta2,
            "tempcorr2": tempcorr2,
            "temp_short_vol2": temp_short_vol2,
            "tempMax2": tempMax2
        }).reset_index(drop=True)  # Reset the index to avoid duplications
        
        # === CONTINUATION: Generate rets ===
        
        # all the xs and y do not have index but with an ID column
        
        # the prices here are not used as X since it is based on to the new rebdate. It is redundant but it is the way the original code was written
        allprices = df_price.copy()
        allprices.columns = ["ID"] + [f"x{i}" for i in range(1, df_price.shape[1])]
        allpricesd = df_daily.copy()
        allpricesd.columns = ["ID"] + [f"xd{i}" for i in range(1, df_daily.shape[1])]
        # allpricesd_old.columns = ["ID"] + [f"xd_old{i}" for i in range(1, df_old.shape[1])]

        # cumulative rets
        df_old.reset_index(drop=False, inplace=True)
        allretsd_old = 1 / (df_old.iloc[:, 2:].div(df_old.iloc[:, 1], axis=0))
        allretsd_old["ID"] = df_old.iloc[:, 0].values
        
        df_daily.reset_index(drop=False, inplace=True)
        allretsd_new = 1 / (df_daily.iloc[:, 2:].div(df_daily.iloc[:, 1], axis=0))
        allretsd_new["ID"] = df_daily.iloc[:, 0].values

        df_price.reset_index(drop=False, inplace=True)
        allrets = 1 / (df_price.iloc[:, 2:].div(df_price.iloc[:, 1], axis=0))
        allrets["ID"] = df_price.iloc[:, 0].values

        allrets_old = 1 / (df_price.iloc[:, 3:].div(df_price.iloc[:, 2], axis=0))
        allrets_old["ID"] = df_price.iloc[:, 0].values

        # define X and y
        y = allrets.iloc[:, [0]].copy()
        y.columns = ["y"]
        y["ID"] = allrets["ID"].values
        y.reset_index(drop=True, inplace=True)

        x1 = allrets_old.iloc[:, :-1].copy()
        x1.columns = [f"m{j+1}" for j in range(x1.shape[1])]
        x1["ID"] = allrets_old["ID"].values
        x1.reset_index(drop=True, inplace=True)

        x2 = allretsd_old.iloc[:, :20].copy()
        x2.columns = [f"d{j+1}" for j in range(20)]
        x2["ID"] = allretsd_old["ID"].values
        x2.reset_index(drop=True, inplace=True)

        xs = pd.merge(x1, x2, on="ID")
        xs = pd.merge(xs, allpricedfactor_old, on="ID")

        # standardize y
        y["classy"] = (y["y"] - y["y"].mean()) / y["y"].std()

        currentdata = pd.merge(y, xs, on="ID")
        currentdata["Date"] = actiondates[i - 1]

        return currentdata
    
    # --- Parallel execution ---
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(worker_func)(i, actiondates, alldata, lookback)
        for i in range(lookback + 5, len(actiondates))
    )

    # --- Combine results (like `.combine = rbind`) ---
    results_df = pd.concat(results, ignore_index=True)
    
   
    filename = f"{pathname}/SHSZ_{actdays}_EODtestraw.parquet"
    results_df.to_parquet(filename)


# %%
