import numpy as np
import pandas as pd

def get_indicator_data(df):
    # Simple moving average
    """
    SMA=Sum of closing price of n periods/n 
    where n is the window size
    """
    df['SMA5'] = df['close'].rolling(window=5).mean()
    df['SMA10'] = df['close'].rolling(window=10).mean()
    df['SMA15'] = df['close'].rolling(window=15).mean()
    df['SMA20'] = df['close'].rolling(window=20).mean()
    
    # Exponential moving averages
    """
    Gives more weight to recent closing prices than older prices
    EMAt=k*Pt+(1-k)*EMAt-1
    """
    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA15'] = df['close'].ewm(span=15, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Bollinger Bands
    """
    Measure volatility by plotting three separate bands. Narrow bands correspond to low volatility
    whereas wide bands correspond to high volatility
    Middle band: SMA of a specified window
    Upper band= Middle band + 2*standard_deviation
    Lower band= Middle band - 2*standard_deviation
    """
    df['BB_MIDDLE'] = df['close'].rolling(window=20).mean()
    df['BB_UPPER'] = df['BB_MIDDLE'] + 2 * df['close'].rolling(window=20).std()
    df['BB_LOWER'] = df['BB_MIDDLE'] - 2 * df['close'].rolling(window=20).std()
    
    # Kaufman adaptive moving averages
    """
    Based on the idea of market volatility
    KAMA= KAMA_prev + SC(Price-KAMA_prev)
    where SC is the smoothening constant which measures the sensitivity to price change
    """
    df['KAMA10'] = df['close'].rolling(window=10).mean()
    df['KAMA20'] = df['close'].rolling(window=20).mean()
   
    # Parabolic stop and reverse
    """
    It is used to calculate the reversal points in a stock price trend
    SAR=SAR_prev + AF(EP-SAR_prev) -- for uptrend
    SAR=SAR_prev - AF(SAR_prev-EP) -- for down trend
    AF-> acceleration factor
    EP-> extreme point
    """
    # Set acceleration factor to 0.02
    # Set maximum acceleration factor to 0.2
    df['SAR'] = df['close'].rolling(window=2).mean() 

    # Triangular moving averages
    """
    TRIMA assigns more weight to the middle portion of a data series
    """
    df['TRIMA5'] = df['close'].rolling(window=5).mean()
    df['TRIMA10'] = df['close'].rolling(window=10).mean()

    # Average directional index
    """
    +DM-> positive directional movement
    -DM-> negative directional movement
    TR -> true range
    After calculating +DM, -DM and TR, the directional indicators are calculated using EMA.
    ADX is the smoothened average of the absolute difference between +DI and -DI
    """
    df['ADX5'] = df['close'].rolling(window=5).mean()
    df['ADX10'] = df['close'].rolling(window=10).mean()
    df['ADX20'] = df['close'].rolling(window=20).mean()

    # Commodity channel index
    """
    Used to idenitfy cyclical trends in markets
    Typical price TP= close+low+high/3
    Calculate SMA for a specified period
    MD->The mean deviation of the SMA and TP for the specified period 
    CCI=(TP-SMA)/(0.015*MD) 
    """
    df['CCI5'] = df['close'].rolling(window=5).mean()
    df['CCI10'] = df['close'].rolling(window=10).mean()
    df['CCI15'] = df['close'].rolling(window=15).mean()

    # Moving average convergence divergence
    """
    Fast period-> corresponds to the window size for the calculation of short-term EMA 
    Slow period-> corresponds to the window size for the calculation of long-term EMA
    The difference between these two gives the MACD line
    EMA of the MACD line gives the signal line(signal  to buy or sells)
    """
    df['MACD510'] = df['close'].ewm(span=5, adjust=False).mean() - df['close'].ewm(span=10, adjust=False).mean()
    df['MACD520'] = df['close'].ewm(span=5, adjust=False).mean() - df['close'].ewm(span=20, adjust=False).mean()
    df['MACD1020'] = df['close'].ewm(span=10, adjust=False).mean() - df['close'].ewm(span=20, adjust=False).mean()
    df['MACD1520'] = df['close'].ewm(span=15, adjust=False).mean() - df['close'].ewm(span=20, adjust=False).mean()
    df['MACD1226'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()

    # Momentum indicator
    """
    Difference between the current closing price and the closing price n periods ago
    where n is the window size
    """
    df['MOM10'] = df['close'].diff(periods=10)
    df['MOM15'] = df['close'].diff(periods=15)
    df['MOM20'] = df['close'].diff(periods=20)

    # Rate of change
    """
    ROC= (Current closing price-closing price n periods ago)/closing price n periods ago
    """
    df['ROC5'] = df['close'].pct_change(periods=5)
    df['ROC10'] = df['close'].pct_change(periods=10)
    df['ROC20'] = df['close'].pct_change(periods=20)

    # Percentage price oscillator
    """
    PPO=((short term moving average-long term moving average)/long term moving avg)*100
    Signal line-> SMA of PPO line produces the signal line
    """
    df['PPO'] = ((df['close'].rolling(window=5).mean() - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).mean()) * 100

    # Relative strength index
    """
    RSI=100-(100/(1+RS))
    here RS or relative strength is the avg of upward price changes divided by the 
    avg of downward price changes for a specified period
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI14'] = 100 - (100 / (1 + RS))
    df['RSI8'] = 100 - (100 / (1 + RS.rolling(window=8).mean()))

    # Stochastic oscillator indicators
    """
    %K line: (Closing price-lowest low)/(Highest high-lowest low) * 100 over a specified period
    %D line: SMA of K line over a given period usually 3
    """
    highest_high = df['high'].rolling(window=14).max()
    lowest_low = df['low'].rolling(window=14).min()
    df['fastk'] = ((df['close'] - lowest_low) / (highest_high - lowest_low)) * 100
    df['fastd'] = df['fastk'].rolling(window=3).mean()

    # Ultimate oscillator(ULTOSC)
    """
    Captures momentum across three different timeframes
    UO=[(4 * Short Average BP) + (2 * Medium Average BP) + (Long Average BP)] /
     [(4 * Short Average TR) + (2 * Medium Average TR) + (Long Average TR)]
     where BP is the buying pressure -> difference between today's close and the minimum of today's low
     or yesterday's close 
    """
    BP = df['close'] - pd.concat([df['low'], df['close'].shift()], axis=1).min(axis=1)
    TR = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    UO = ((BP.rolling(window=7).sum() / TR.rolling(window=7).sum()) * 4 +
          (BP.rolling(window=14).sum() / TR.rolling(window=14).sum()) * 2 +
          (BP.rolling(window=28).sum() / TR.rolling(window=28).sum())) / 7
    df['ULTOSC'] = UO

    # William's %R (WILLR) 14 day period
    """
    %R= (highest high-close)/(highest high-lowest low) * -100
    """
    highest_high = df['high'].rolling(window=14).max()
    lowest_low = df['low'].rolling(window=14).min()
    df['WILLR'] = ((highest_high - df['close']) / (highest_high - lowest_low)) * -100

    # Average true range
    """
    ATR=SMA(TR,n)/n
    """
    TR = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    df['ATR7'] = TR.rolling(window=7).mean()
    df['ATR14'] = TR.rolling(window=14).mean()

    # True range
    """
    TR= max(high-low,|high-previous close|,|low-previous close|)
    """
    df['Trange'] = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)

    # Typical price
    """
    TP= (high+low+close)/3
    """
    df['TYPPRICE'] = (df['high'] + df['low'] + df['close']) / 3

    # Vortex indicators (used to identify a new trend or the direction of an ongoing trend)
    """
    +VI= current high - previous low
    -VI= current low - previous high
    """
    df['VIn'] = abs(df['high'] - df['low'].shift())
    df['VIp'] = abs(df['low'] - df['high'].shift())

    # Money flow volume
    """
    Money flow= Typical price*volume
    """
    if 'volume' in df.columns:
        df['MFV'] = df['TYPPRICE'] * df['volume']

    del df['open']
    del df['high']
    del df['low']   
    return df


def exponential_smooth(data, alpha):
    """
    This function performs exponential smoothing on a dataset 
    to reduce noise and make values less rigid. Exponential 
    smoothing is a technique commonly used in time series analysis 
    to assign exponentially decreasing weights to past observations.
    """
    return data.ewm(alpha=alpha).mean()

def make_prediction(data, lookahead):
    """
    Function to produce truth values
    At a given row, it moves the dataset by the window parameter
    in backward direction to see if the closing price increased (1) or decreased (0)
    """
    
    prediction = (data.shift(-lookahead)['close'] >= data['close'])
    prediction = prediction.iloc[:-lookahead]
    data['pred'] = prediction.astype(int)
    return data
"""
df_main = pd.read_csv('../../data/reversed_NIFTY_50_23_years.csv') 
df_main = get_indicator_data(df_main)
df_main.to_csv('final_data.csv', index=False)  
"""