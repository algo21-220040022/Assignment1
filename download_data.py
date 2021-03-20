import ccxt
import pandas as pd
import datetime
import numpy as np
import pickle
import time
epoch = datetime.datetime.utcfromtimestamp(0)


def unix_time_millis(dt):
    if isinstance(dt, str):
        dt=pd.to_datetime(dt)
    return (dt - epoch).total_seconds() * 1000.0

def utc_to_strftime(utc_timestamp):
    return datetime.datetime.fromtimestamp(utc_timestamp/1000).strftime("%Y-%m-%d %H:%M:%S")

exchange = ccxt.bittrex()

exchange.fetch_trades('BTC/USDT')

date = "2016-12-31"
last_year = date[:4]
time_keeper = 1
try:
    with open(f"BTC_min_{last_year}.pkl", "rb") as f:
        all_daily_data = pickle.load(f)
    last_read_date = str(all_daily_data.index[0])[:10]
    date = str(pd.to_datetime(last_read_date) - datetime.timedelta(days=1))[:10]
except:
    all_daily_data = pd.DataFrame()
while True:
    if date[:4] != last_year:
        print(f'Finish to load year {last_year} data to BTC_min_{last_year}.pkl....\n')
        last_year = date[:4]
        all_daily_data = pd.DataFrame()
        time_keeper = 1
        time.sleep(5)

    print(f'Download data on {date}...')
    daily_data = exchange.fetch_ohlcv('BTC/USDT', '1m', since=unix_time_millis(date), limit=1440)
    daily_data_df =pd.DataFrame(np.array(daily_data), columns=["utc_epoch","open","high","low","close","volume"])
    daily_data_df.index = daily_data_df['utc_epoch'].apply(utc_to_strftime)
    all_daily_data = pd.concat([daily_data_df, all_daily_data], axis=0)
    date = str(pd.to_datetime(date) - datetime.timedelta(days=1))[:10]
    if time_keeper%30 == 0 or date[:4] != last_year:
        print(f"Tmp dump data to BTC_min_{last_year}.pkl")
        all_daily_data.sort_index()
        file = open(f"BTC_min_{last_year}.pkl", "wb")
        pickle.dump(all_daily_data, file)
        file.close()
    time.sleep(np.random.uniform(0,1))
    time_keeper += 1
