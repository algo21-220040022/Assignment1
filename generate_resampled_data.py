import pandas as pd
import pickle

"""Load raw data"""
all_data = pd.DataFrame()
for year in [str(y) for y in range(2016, 2022)]:
    with open(f"BTC_min_{year}.pkl", "rb") as f:
        data_tmp = pickle.load(f)
    all_data = pd.concat([all_data, data_tmp])
all_data.rename_axis("date", inplace=True)
all_data.sort_values(by="utc_epoch", inplace=True)
all_data.index = pd.to_datetime(all_data.index)

"""Calculate absolute percentage change"""
all_data['abs_pct'] = abs(all_data['close'].pct_change())
all_data.dropna(inplace=True)

"""Generate resampled data"""
variation_threshold = 0.02
cum_abs_change = 0
is_start = True
start_dt = 0

open_list = []
high_list = []
low_list = []
close_list = []
volume_list = []
dt_list = []

for dt in all_data.index:
    print(f"{dt}....")
    if is_start:
        start_dt = dt
        is_start = False
    cum_abs_change += all_data.at[dt, "abs_pct"]
    if cum_abs_change > 0.02:
        open_list.append(all_data.at[start_dt, "open"])
        high_list.append(all_data.loc[(all_data.index >= start_dt) & (all_data.index <= dt)]['high'].max())
        low_list.append(all_data.loc[(all_data.index >= start_dt) & (all_data.index <= dt)]['low'].min())
        close_list.append(all_data.at[dt, "close"])
        volume_list.append(all_data.loc[(all_data.index >= start_dt) & (all_data.index <= dt)]['high'].sum())
        dt_list.append(dt)
        print(f"cum_abs_change={cum_abs_change}; open={open_list[-1]}; high={high_list[-1]}"
              f"low={low_list[-1]}; close={close_list[-1]}; volume={volume_list[-1]}")
        cum_abs_change = 0
        is_start = True

resampled_data = pd.DataFrame({"open": open_list, "high": high_list, "low": low_list,
                               "close": close_list, "volume": volume_list}, index=dt_list)

with open("BTC_resampled_data.pkl", "wb") as f:
    pickle.dump(resampled_data, f)

from datetime import datetime
resampled_data = pickle.load(open("BTC_resampled_data.pkl", "rb"))
bit_difficulty = pd.read_csv("btc.com_diff_2021-03-17_02_48_51.csv")
bit_difficulty.index = bit_difficulty['timestamp'].apply(datetime.fromtimestamp)

diff_list = []
ave_hashrate = []
ave_time = []
diff_change = []
for dt in resampled_data.index:
    latest_bit_difficulty = bit_difficulty.loc[bit_difficulty.index <= dt].iloc[-1]
    diff_list.append(latest_bit_difficulty['diff'])
    ave_hashrate.append(latest_bit_difficulty['average_hashrate'])
    ave_time.append(latest_bit_difficulty['average_time(seconds)'])
    diff_change.append(latest_bit_difficulty['change'])
    print(f'{dt}: diff={diff_list[-1]}; ave_hashrate={ave_hashrate[-1]}; ave_time={ave_time[-1]}; diff_change={diff_change[-1]}')
resampled_data['diff'] = diff_list
resampled_data['ave_hashrate'] = ave_hashrate
resampled_data['ave_time'] = ave_time
resampled_data['diff_change'] = diff_change
with open("BTC_resampled_data_and_ribbon.pkl", "wb") as f:
    pickle.dump(resampled_data, f)