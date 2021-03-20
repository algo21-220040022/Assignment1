# predict-bitcoin-by-ML
The purpose of this project is to predict the bitcoin price movement by machine learning. The general process is as follow:

## 1. download_data.py
This part is used to down load the historical monthly bicoin price data. Notice that there is some issues to continue download data for the package or exchange would limit your call frequency. But don't worry about that. The program would output the data to .pkl file in working directory for a certain frequency. Every time the program is terminated by "Time out" error, you only need to restart the program and change the variable "last_year" to the year last time you stop(in other word, you can see the working directory and use the smallest year among .pkl files as "last_year").

## 2. generate_resampled_data.py
This part refers to article **_Ensemble of machine learning algorithms for cryptocurrency investment with different data resampling methods_**. It transform the original bar data to new bar by resampling and aggregate several continuous bars. The resampling is based on the cumulative absolute percentage change. That is, we start from one bar to subsequent bars and calculate cumulative absolute percentage changes. When the cumulative number exceed 2%, we stop and use the bar you pass so far to construct a single new bar.

## 3. generate_technical_indicators.py
This part is used to generate some technical indicators from the resampled data. I extend the reference to contain much more indicators, including the difficulty and average time of bitcoin mining. I also calculate the lag return and the other technical indicators(more than on hundred) from package talib. Besides, I also construct a wavelet function to filter the data, but it doesn't work as what I expected and thus I don't utilize in my data proccessing.

## 4. generate_label.py
This part is used to generate the label data for the supervised learning. I denote the label of a sample by 1 if its forward return is positive and 0 otherwise.

## 5. train_and_test_model_by_ML.py
This part is to conduct the prediction by ML. Different from the article, I conduct the feature selection first before I develop my models and my comparison experiment show that this selection can significantly increase the prediciton power of all the models. Then I use random forest, XGBoost, Logistic regression and LSTM(which is not mention in the reference) to predict. The result shows that all the methods can not achieve a prediction accuracy more than 60%. Even the LSTM can not beat the other models in my effort.
