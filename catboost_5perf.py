# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 20:57:03 2021

@author: windwine5900
"""

#%% read the feather data
#%% read the feather data
import os
import pandas as pd
import numpy as np
import time
from datetime import date
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import optuna
from sklearn.utils import resample
from lightgbm import LGBMRegressor
from sklearn.utils import resample
import catboost as cat
from catboost import Pool
import polars as pl
import matplotlib.pyplot as plt
# import pyfolio as pf

# def perf_plot():
#     pf.create_full_tear_sheet(port)
#     # calculate annualized return
#     annual_return = pf.timeseries.annual_return(port, period='weekly')
#     # calculate maximum drawdown
#     max_drawdown = pf.timeseries.gen_drawdown_table(port).iloc[0, 0]/100
#     # calculate sharpe ratio
#     sharpe_ratio = pf.timeseries.sharpe_ratio(port, period='weekly')
#     calmar_ratio = pf.timeseries.calmar_ratio(port)
#     calmar_ratio = annual_return/max_drawdown

#     # print results
#     print(f"Annualized return: {annual_return:.2%}")
#     print(f"Maximum drawdown: {max_drawdown:.2%}")
#     print(f"Sharpe ratio: {sharpe_ratio:.2f}")
#     print(f"Calmar ratio: {calmar_ratio:.2f}")

import warnings
warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore', category=DeprecationWarning)




#%%


os.chdir(r'e:/laosongdata')

for actdays in np.arange(1,6):
    input_filename = f'SHSZ_{actdays}_EODtestraw.parquet'
 
    print(input_filename)
    
    MLdata = pd.read_parquet(input_filename) # the data is w/o index, the data is already Z scored on classy and fillna with 0 for x

    print(MLdata.columns)
    
    MLdata['Date'] = pd.to_datetime(MLdata['Date'])
    MLdata['Janind'] = MLdata['Date'].apply(lambda x: x.month)
    
    # Define the desired column order
    desired_order = ['ID', 'Date', 'y', 'classy']

    # Get the remaining columns that are not in the desired order
    remaining_columns = [col for col in MLdata.columns if col not in desired_order]

    # Rearrange the DataFrame columns
    MLdata = MLdata[desired_order + remaining_columns]
    ###########################################
    # x_indexs = np.r_[9:56] # M:zscore  10:56 in the r convention
    x_indexs = np.arange(4, MLdata.shape[1])  # 4:end of columns
    print(MLdata.columns[x_indexs])
    y_name = "classy"

    cutofftime = pd.Timestamp('2016-01-01')

    train = MLdata[MLdata.Date<cutofftime]
    test = MLdata[MLdata.Date>=cutofftime]

    Ytrain = train[y_name].values
    Xtrain = train.iloc[:,x_indexs].values
    
    # train the default model

    model = cat.CatBoostRegressor(
            logging_level="Silent",
            iterations=5000,
            task_type = 'GPU'
        )

    train_pool = Pool(Xtrain, Ytrain)
    model.fit(train_pool)
    
    # go with the test set
    Xtest = test.iloc[:,x_indexs].values
    Ytest = test[y_name].values
    test_pool = Pool(Xtest, Ytest)   
    Z = model.predict(test_pool)

    pred_y = Z
    bigtest = test.copy() # just my naming convention to get the pred_y into the table
    bigtest['pred_y'] = pred_y

    filename = 'catboost' + input_filename
    # bigtest.reset_index().to_feather(filename) # bridge to R
    # filename = 'catboost' + 'SHSZ_'+ str(actdays) +'.parquet'
    bigtest.reset_index().to_parquet(filename,engine="pyarrow") # bridge to R
    
    # Assuming 'bigtest' is already your DataFrame
    # Ensure 'Date' is in datetime format and set as index along with 'ID'
    bigtest['Date'] = pd.to_datetime(bigtest['Date'])
    bigtest['y'] = bigtest['y']-1
    bigtest.set_index(['Date', 'ID'], inplace=True)

    # Sort by 'pred_y' within each 'Date'
    bigtest.sort_values(by=['Date', 'pred_y'], ascending=[True, False], inplace=True)

    ngroup = 10
    # Calculate decile ranks
    bigtest['decile'] = bigtest.groupby('Date')['pred_y'].transform(
        lambda x: pd.qcut(x, ngroup, labels=False, duplicates='drop')
    )

    # Determine the deciles for long and short positions
    # Top 10% (decile 9) and bottom 10% (decile 0)
    long_decile = ngroup-1
    short_decile = 0

    # Filter data for long and short positions
    longs = bigtest[bigtest['decile'] == long_decile]
    shorts = bigtest[bigtest['decile'] == short_decile]

    # Calculate daily returns for long and short positions
    # Assume 'y' is the actual return column
    daily_returns = bigtest.groupby(['Date', 'decile'])['y'].mean()
    long_returns = daily_returns.xs(long_decile, level='decile')
    short_returns = daily_returns.xs(short_decile, level='decile')

    # Calculate strategy returns (long - short)
    strategy_returns = long_returns - short_returns

    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.suptitle(f'Portfolio Analysis {filename}', fontsize=16)

    # Decile plot
    plt.subplot(1, 2, 1)
    bigtest.groupby('decile')['y'].mean().plot(kind='bar')
    plt.title('Average y by Decile')
    plt.xlabel('Decile')
    plt.ylabel('Average y')

    # Cumulative returns plot
    plt.subplot(1, 2, 2)
    cumulative_returns.plot()
    plt.title('Cumulative Portfolio Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')

    plt.tight_layout()
    plt.show()

    


#%%














# %%
