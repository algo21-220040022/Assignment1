import pywt
import pandas as pd

import talib  # calculate technical indicator
import pickle

# 1. denoise the data by wavelet
def wt_denoise(data, threshold=0.4, wavelet_type = 'db8'):
    """
    denoise data through wavelet transform
    :param data:
    :param threshold: threshold to filter. The larger it is, the more smooth data would be
    :param wavelet_type:
    :return: denoised data
    """
    w = pywt.Wavelet(wavelet_type)  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    print("maximum level is " + str(maxlev))

    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(data, wavelet_type, level=maxlev)  # 将信号进行小波分解

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波

    datarec = pywt.waverec(coeffs, wavelet_type)  # 将信号进行小波重构
    return datarec

# 2. get all the additional variables(technical indices and others based on the original OCLH data and volumn data)
def get_exogenous_data(data,**kwargs):
    """
    :param data: DataFrame of underlying market info, should include ['open', 'close', 'low', 'high', 'volumn', 'money'] columns, where 'money' is trading amount.
    :param kwargs: parameter for construction of new variables
    :return: dict of the addictional varibales
    """
    open = data['open']
    high = data['high']
    low = data['low']
    close = data['close']
    volume = data['volume']
    all_para = kwargs.keys()  # may be useful in the future??
    indicators_df = pd.DataFrame()

    """
    get all the technical indicators, reference: https://mrjbq7.github.io/ta-lib/doc_index.html
    """
    """momentum indicators"""
    indicators_df['ADX'] = talib.ADX(high, low, close, timeperiod=14)  # Average Directional Movement Index
    indicators_df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)  # Average Directional Movement Index Rating
    indicators_df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)  # Absolute Price Oscillator
    indicators_df['aroondown'], indicators_df['aroonup'] = talib.AROON(high, low, timeperiod=14)  # Aroon
    indicators_df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)  # Aroon Oscillator
    indicators_df['BOP'] = talib.BOP(open, high, low, close)  # Balance Of Power
    indicators_df['CCI'] = talib.CCI(high, low, close, timeperiod=14)  # Commodity Channel Index
    indicators_df['CMO'] = talib.CMO(close, timeperiod=14)  # Chande Momentum Oscillator
    indicators_df['DX'] = talib.DX(high, low, close, timeperiod=14)  # Directional Movement Index
    indicators_df['macd1'], indicators_df['macdsignal1'], macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)  # Moving Average Convergence/Divergence
    indicators_df['macd2'], indicators_df['macdsignal2'], macdhist = talib.MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0,signalperiod=9, signalmatype=0)  # MACD with controllable MA type
    indicators_df['macd3'], indicators_df['macdsignal3'], macdhist = talib.MACDFIX(close, signalperiod=9)  # Moving Average Convergence/Divergence Fix 12/26
    indicators_df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)  # Money Flow Index
    indicators_df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)  # Minus Directional Indicator
    indicators_df['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)  # Minus Directional Movement
    indicators_df['MOM'] = talib.MOM(close, timeperiod=10)  # Momentum
    indicators_df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)  # Plus Directional Indicator
    indicators_df['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)  # Plus Directional Movement
    indicators_df['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)  # Percentage Price Oscillator
    indicators_df['ROC'] = talib.ROC(close, timeperiod=10)  # Rate of change : ((price/prevPrice)-1)*100
    indicators_df['ROCP'] = talib.ROCP(close, timeperiod=10)  # Rate of change Percentage: (price-prevPrice)/prevPrice
    indicators_df['ROCR'] = talib.ROCR(close, timeperiod=10)  # Rate of change ratio: (price/prevPrice)
    indicators_df['ROCR100'] = talib.ROCR100(close, timeperiod=10)  # Rate of change ratio 100 scale: (price/prevPrice)*100
    indicators_df['RSI'] = talib.RSI(close, timeperiod=14)  # Relative Strength Index
    indicators_df['slowk'], indicators_df['slowd'] = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)  # Stochastic
    indicators_df['fastk'], indicators_df['fastd'] = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)  # Stochastic Fast
    indicators_df['S-RSIk'], indicators_df['S-RSId'] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)  # Stochastic Relative Strength Index
    indicators_df['TRIX'] = talib.TRIX(close, timeperiod=30)  # 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    indicators_df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)  # Ultimate Oscillator
    indicators_df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)  # Williams' %R

    """Volume indicators"""
    indicators_df['AD'] = talib.AD(high, low, close, volume)  # Chaikin A/D Line
    indicators_df['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)  # Chaikin A/D Oscillator
    indicators_df['OBV'] = talib.OBV(close, volume)  # On Balance Volume

    """Volatility indicators"""
    indicators_df['ATR'] = talib.ATR(high, low, close, timeperiod=14)  # Average True Range
    indicators_df['NATR'] = talib.NATR(high, low, close, timeperiod=14)  # Normalized Average True Range
    indicators_df['TRANGE'] = talib.TRANGE(high, low, close)  # True Range

    """Cycle Indicators"""
    indicators_df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)  # Hilbert Transform - Dominant Cycle Period
    indicators_df['HT_DCPHASE'] = talib.HT_DCPHASE(close)  # Hilbert Transform - Dominant Cycle Phase
    indicators_df['inphase'], indicators_df['quadrature'] = talib.HT_PHASOR(close)  # Hilbert Transform - Phasor Components
    indicators_df['sine'], indicators_df['leadsine'] = talib.HT_SINE(close)  # Hilbert Transform - SineWave
    indicators_df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)  # Hilbert Transform - Trend vs Cycle Mode

    """Pattern Recognition"""
    indicators_df['CDL2CROWS'] = talib.CDL2CROWS(open, high, low, close)  # Two Crows
    indicators_df['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(open, high, low, close)  # Three Black Crows
    indicators_df['CDL3INSIDE'] = talib.CDL3INSIDE(open, high, low, close)  # Three Inside Up/Down
    indicators_df['CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(open, high, low, close)  # Three-Line Strike
    indicators_df['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(open, high, low, close)  # Three Outside Up/Down
    indicators_df['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(open, high, low, close)  # Three Stars In The South
    indicators_df['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(open, high, low, close)  # Three Advancing White Soldiers
    indicators_df['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(open, high, low, close, penetration=0)  # Abandoned Baby
    indicators_df['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(open, high, low, close)  # Advance Block
    indicators_df['CDLBELTHOLD'] = talib.CDLBELTHOLD(open, high, low, close)  # Belt-hold
    indicators_df['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(open, high, low, close)  # Breakaway
    indicators_df['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(open, high, low, close)  # Closing Marubozu
    indicators_df['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(open, high, low, close)  # Concealing Baby Swallow
    indicators_df['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(open, high, low, close)  # Counterattack
    indicators_df['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(open, high, low, close, penetration=0)  # Dark Cloud Cover
    indicators_df['CDLDOJI'] = talib.CDLDOJI(open, high, low, close)  # Doji
    indicators_df['CDLDOJISTAR'] = talib.CDLDOJISTAR(open, high, low, close)  # Doji Star
    indicators_df['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(open, high, low, close)  # Dragonfly Doji
    indicators_df['CDLENGULFING'] = talib.CDLENGULFING(open, high, low, close)  # Engulfing Pattern
    indicators_df['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(open, high, low, close, penetration=0)  # Evening Doji Star
    indicators_df['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(open, high, low, close, penetration=0)  # Evening Star
    indicators_df['CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(open, high, low, close)  # Up/Down-gap side-by-side white lines
    indicators_df['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(open, high, low, close)  # Gravestone Doji
    indicators_df['CDLHAMMER'] = talib.CDLHAMMER(open, high, low, close)  # Hammer
    indicators_df['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(open, high, low, close)  # Hanging Man
    indicators_df['CDLHARAMI'] = talib.CDLHARAMI(open, high, low, close)  # Harami Pattern
    indicators_df['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(open, high, low, close)  # Harami Cross Pattern
    indicators_df['CDLHIGHWAVE'] = talib.CDLHIGHWAVE(open, high, low, close)  # High-Wave Candle
    indicators_df['CDLHIKKAKE'] = talib.CDLHIKKAKE(open, high, low, close)  # Hikkake Pattern
    indicators_df['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(open, high, low, close)  # Modified Hikkake Pattern
    indicators_df['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(open, high, low, close)  # Homing Pigeon
    indicators_df['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(open, high, low, close)  # Identical Three Crows
    indicators_df['CDLINNECK'] = talib.CDLINNECK(open, high, low, close)  # In-Neck Pattern
    indicators_df['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(open, high, low, close)  # Inverted Hammer
    indicators_df['CDLKICKING'] = talib.CDLKICKING(open, high, low, close)  # Kicking
    indicators_df['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(open, high, low, close)  # Kicking - bull/bear determined by the longer marubozu
    indicators_df['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(open, high, low, close)  # Ladder Bottom
    indicators_df['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(open, high, low, close)  # Long Legged Doji
    indicators_df['CDLLONGLINE'] = talib.CDLLONGLINE(open, high, low, close)  # Long Line Candle
    indicators_df['CDLMARUBOZU'] = talib.CDLMARUBOZU(open, high, low, close)  # Marubozu
    indicators_df['CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(open, high, low, close)  # Matching Low
    indicators_df['CDLMATHOLD'] = talib.CDLMATHOLD(open, high, low, close, penetration=0)  # Mat Hold
    indicators_df['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(open, high, low, close, penetration=0)  # Morning Doji Star
    indicators_df['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(open, high, low, close, penetration=0)  # Morning Star
    indicators_df['CDLONNECK'] = talib.CDLONNECK(open, high, low, close)  # On-Neck Pattern
    indicators_df['CDLPIERCING'] = talib.CDLPIERCING(open, high, low, close)  # Piercing Pattern
    indicators_df['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(open, high, low, close)  # Rickshaw Man
    indicators_df['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(open, high, low, close)  # Rising/Falling Three Methods
    indicators_df['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(open, high, low, close)  # Separating Lines
    indicators_df['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(open, high, low, close)  # Shooting Star
    indicators_df['CDLSHORTLINE'] = talib.CDLSHORTLINE(open, high, low, close)  # Short Line Candle
    indicators_df['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(open, high, low, close)  # Spinning Top
    indicators_df['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(open, high, low, close)  # Stalled Pattern
    indicators_df['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(open, high, low, close)  # Stick Sandwich
    indicators_df['CDLTAKURI'] = talib.CDLTAKURI(open, high, low, close)  # Takuri (Dragonfly Doji with very long lower shadow)
    indicators_df['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(open, high, low, close)  # Tasuki Gap
    indicators_df['CDLTHRUSTING'] = talib.CDLTHRUSTING(open, high, low, close)  # Thrusting Pattern
    indicators_df['CDLTRISTAR'] = talib.CDLTRISTAR(open, high, low, close)  # Tristar Pattern
    indicators_df['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(open, high, low, close)  # Unique 3 River
    indicators_df['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(open, high, low, close)  # Upside Gap Two Crows
    indicators_df['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(open, high, low, close)  # Upside/Downside Gap Three Methods

    return indicators_df


BTC_data = pickle.load(open("BTC_resampled_data_and_ribbon.pkl", "rb"))
BTC_data['ret'] = BTC_data['close'].pct_change().shift(-1)
BTC_data['ret_lag1'] = BTC_data['ret'].shift(1)
BTC_data['ret_lag2'] = BTC_data['ret'].shift(2)
BTC_data['ret_lag3'] = BTC_data['ret'].shift(3)
BTC_data['ret_lag4'] = BTC_data['ret'].shift(4)
BTC_data['ret_lag5'] = BTC_data['ret'].shift(5)
BTC_data['ret_lag6'] = BTC_data['ret'].shift(6)
BTC_data['ret_lag7'] = BTC_data['ret'].shift(7)
BTC_data['ret_lag8'] = BTC_data['ret'].shift(8)
BTC_data['ret_lag9'] = BTC_data['ret'].shift(9)
BTC_data['ret_lag10'] = BTC_data['ret'].shift(10)
#BTC_data['ret'] = BTC_data['close'].pct_change(periods=5).shift(-5)


technical_indicators = get_exogenous_data(BTC_data)
BTC_data = pd.concat([BTC_data, technical_indicators], axis=1).dropna()


with open("BTC_resampled_data_and_ribbon_tech.pkl", "wb") as f:
    pickle.dump(BTC_data, f)
