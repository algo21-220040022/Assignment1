import pickle

threshold = 0
BTC_data = pickle.load(open("BTC_resampled_data_and_ribbon_tech.pkl", "rb"))
label_list = []
for dt in BTC_data.index:
    ret_dt = BTC_data.at[dt, "ret"]
    if ret_dt > threshold:
        label_list.append(1)
    elif ret_dt <= threshold:
        label_list.append(0)
    else:
        label_list.append(0)
BTC_data['Y'] = label_list
with open("BTC_resampled_data_and_ribbon_tech_label.pkl", "wb") as f:
    pickle.dump(BTC_data, f)

BTC_data = pickle.load(open("BTC_resampled_data_and_ribbon_tech_ret5.pkl", "rb"))
label_list = []
for dt in BTC_data.index:
    ret_dt = BTC_data.at[dt, "ret"]
    if ret_dt > threshold:
        label_list.append(1)
    elif ret_dt <= -threshold:
        label_list.append(-1)
    else:
        label_list.append(0)
BTC_data['Y'] = label_list
with open("BTC_resampled_data_and_ribbon_tech_label_ret5.pkl", "wb") as f:
    pickle.dump(BTC_data, f)