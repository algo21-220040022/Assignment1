from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def min_max_std(_train_data, _test_data):
    indx = _train_data.columns.get_loc('ave_time')
    _train_data1 = _train_data.iloc[:, :indx+1]
    _train_data2 = _train_data.iloc[:, indx+1:]
    _train_data1 = (_train_data1 - _train_data1.min())/(_train_data1.max() - _train_data1.min())

    _test_data1 = _test_data.iloc[:, :indx+1]
    _test_data2 = _test_data.iloc[:, indx+1:]
    _test_data1 = (_test_data1 - _train_data1.min()) / (_test_data1.max() - _train_data1.min())

    return pd.concat([_train_data1, _train_data2], axis=1), pd.concat([_test_data1, _test_data2], axis=1)

def choose_most_important_varibales(X, Y, thre=0.0001):
    RFC = RandomForestClassifier(criterion="entropy", random_state=0,
                                 n_estimators=100, min_samples_split=10,
                                 n_jobs=3, oob_score=True)
    RFC.fit(X, Y)
    print(RFC.oob_score_)
    exog_names = X.columns
    exog_importance = [(k, v) for k, v in zip(exog_names, RFC.feature_importances_)]
    exog_importance = sorted(exog_importance, key=lambda t: t[1], reverse=True)
    most_important_varibales = [item[0] for item in exog_importance if (item[1] >= thre and item[0] in X.columns)]
    return most_important_varibales, exog_importance

"""Prepare train and test data"""
BTC_data = pickle.load(open("BTC_resampled_data_and_ribbon_tech_label.pkl", "rb"))
abs_ret = abs(BTC_data['ret'])
BTC_data.drop(columns=['ret'], inplace=True)
BTC_data['ave_hashrate'] = [s[:-11] for s in BTC_data['ave_hashrate'].tolist()]
BTC_data['ave_hashrate'] = BTC_data['ave_hashrate'].astype(int)
train_data = BTC_data.loc[(BTC_data.index >= pd.to_datetime("2020-01-01")) & (BTC_data.index <= pd.to_datetime("2020-12-31"))]
test_data = BTC_data.loc[BTC_data.index >= pd.to_datetime("2021-01-01")]
X_train = train_data.iloc[:, :-1]
Y_train = train_data.iloc[:, -1]
abs_ret_train = abs_ret[X_train.index]
X_test = test_data.iloc[:, :-1]
Y_test = test_data.iloc[:, -1]
X_train, X_test = min_max_std(X_train, X_test)

"""Features selection"""
most_important_features, _ = choose_most_important_varibales(X_train, Y_train, thre=0.001)
X_train_with_most_impt = X_train.loc[:, most_important_features]
X_test_with_most_impt = X_test.loc[:, most_important_features]

"""Random forest"""
RFC = RandomForestClassifier(criterion="entropy",
                             random_state=0,
                             n_estimators=200,
                             n_jobs=3, oob_score=True)
RFC.fit(X_train_with_most_impt, Y_train)
Y_pre = RFC.predict(X_test_with_most_impt)
print(accuracy_score(Y_test, Y_pre))
print(classification_report(Y_test, Y_pre))

"""XGBoost"""
XGBoost = xgb.XGBClassifier(n_estimators=200, max_depth=100, n_jobs=4, gamma=0.01,
                            random_state=10, reg_alpha=0.001, reg_lambda=0.05)
XGBoost.fit(X_train_with_most_impt, Y_train)
Y_pre = XGBoost.predict(X_test_with_most_impt)
print(accuracy_score(Y_test, Y_pre))
print(classification_report(Y_test, Y_pre))

"""Logistic regression"""
LR = LogisticRegression(max_iter=1000, n_jobs=4, random_state=10)
LR.fit(X_train_with_most_impt, Y_train)
Y_pre = LR.predict(X_test_with_most_impt)
print(accuracy_score(Y_test, Y_pre))
print(classification_report(Y_test, Y_pre))

Y_proba_RFC = RFC.predict_proba(X_test_with_most_impt)
Y_proba_XGB = XGBoost.predict_proba(X_test_with_most_impt)
Y_proba_LR = LR.predict_proba(X_test_with_most_impt)
Y_proba = [[(p1[0]+p2[0]+p3[0])/3, (p1[1]+p2[1]+p3[1])/3] for p1,p2,p3 in zip(Y_proba_RFC, Y_proba_XGB, Y_proba_LR)]
Y_pre = [1 if p[1]>0.5 else 0 for p in Y_proba]
print(accuracy_score(Y_test, Y_pre))
print(classification_report(Y_test, Y_pre))

"""LSTM"""
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras.layers import LSTM, Activation
from keras import optimizers


TRAIN_RATIO = 0.8

X_values = X_train_with_most_impt.values
Y_values = Y_train.values
abs_ret_values = abs_ret_train.values
_abs_ret_train = abs_ret_values[:int(X_values.shape[0]*TRAIN_RATIO)] * 50
_X_train = X_values[:int(X_values.shape[0]*TRAIN_RATIO), :]
_Y_train = Y_values[:int(X_values.shape[0]*TRAIN_RATIO)]
_X_valid = X_values[int(X_values.shape[0]*TRAIN_RATIO):, :]
_Y_valid = Y_values[int(X_values.shape[0]*TRAIN_RATIO):]
_X_test = X_test_with_most_impt.values
_Y_test = Y_test.values

_X_train = _X_train.reshape((_X_train.shape[0], 1, _X_train.shape[1]))
_X_valid = _X_valid.reshape((_X_valid.shape[0], 1, _X_valid.shape[1]))
_X_test = _X_test.reshape((_X_test.shape[0], 1, _X_test.shape[1]))
print(_X_train.shape, _Y_train.shape, _X_valid.shape, _Y_valid.shape, _X_test.shape, _Y_test.shape)


model = Sequential()
model.add(LSTM(100,  activation="relu", input_shape=(_X_train.shape[1], _X_train.shape[2]), return_sequences=True))
model.add(Dense(50, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(50, activation="softmax"))
model.add(Dense(50, activation="softmax"))
model.add(Dense(50, activation="softmax"))
model.add(Dense(1, activation="threshold"))
opt = optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss=losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

LSTM_model = model.fit(_X_train, _Y_train, epochs=100, batch_size=300,
                       validation_data=(_X_valid, _Y_valid), verbose=2,
                       sample_weight=_abs_ret_train, shuffle=False)
