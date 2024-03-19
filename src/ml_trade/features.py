"""
High: Highest price at which a security is traded during a trading session
Low: Lowest price at which a security is traded during a trading session
Open: The first price at which a security is traded during a trading session
Close: The last price at which a security is traded during a trading session
"""

import pandas as pd
from finta import TA

# Load the data 
file_path='../../data/reversed_NIFTY_50_23_years.csv'
df=pd.read_csv(file_path)

def get_indicator_data(df):
    # Simple moving average
    """
    SMA=Sum of closing price of n periods/n 
    where n is the window size
    """
    df['SMA5']=TA.SMA(df,5)
    df['SMA10']=TA.SMA(df,10)
    df['SMA15']=TA.SMA(df,15)
    df['SMA20']=TA.SMA(df,20)
    
    # Exponential moving averages
    """
    Gives more weight to recent closing prices than older prices
    EMAt=k*Pt+(1-k)*EMAt-1
    """
    df['EMA5']=TA.EMA(df,5)
    df['EMA10']=TA.EMA(df,10)
    df['EMA15']=TA.EMA(df,15)
    df['EMA20']=TA.EMA(df,20)

    # Bollinger Bands
    """
    Measure volatility by plotting three separate bands. Narrow bands correspond to low volatility
    whereas wide bands correspond to high volatility
    Middle band: SMA of a specified window
    Upper band= Middle band + 2*standard_deviation
    Lower band= Middle band - 2*standard_deviation
    """
    df['upperband']=TA.BBANDS(df)['BB_UPPER']
    df['middleband']=TA.BBANDS(df)['BB_MIDDLE']
    df['lowerband']=TA.BBANDS(df)['BB_LOWER']
    
    # Kaufman adaptive moving averages
    """
    Based on the idea of market volatility
    KAMA= KAMA_prev + SC(Price-KAMA_prev)
    where SC is the smoothening constant which measures the sensitivity to price change
    """
    df['KAMA10']=TA.KAMA(df,10)
    df['KAMA20']=TA.KAMA(df,20)

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
    df['SAR']=TA.SAR(df,0.02,0.2) 

    # Triangular moving averages
    """
    TRIMA assigns more weight to the middle portion of a data series
    """
    df['TRIMA5']=TA.TRIMA(df,5)
    df['TRIMA10']=TA.TRIMA(df,10)

    # Average directional index
    """
    +DM-> positive directional movement
    -DM-> negative directional movement
    TR -> true raange
    After calculating +DM, -DM and TR, the directionaal indicators are calculated using EMA.
    ADX is the smoothened average of the absolute difference between +DI and -DI
    """
    df['ADX5']=TA.ADX(df,5)
    df['ADX10']=TA.ADX(df,10)
    df['ADX20']=TA.ADX(df,20)

    # Commodity channel index
    """
    Used to idenitfy cyclical trends in markets
    Typical price TP= close+low+high/3
    Calculate SMA for a specified period
    MD->The mean deviation of the SMA and TP for the specified period 
    CCI=(TP-SMA)/(0.015*MD) 
    """
    df['CCI5']=TA.CCI(df,5)
    df['CCI10']=TA.CCI(df,10)
    df['CCI15']=TA.CCI(df,15)

    # Moving average convergence divergence
    """
    Fast period-> corresponds to the window size for the calculation of short-term EMA 
    Slow period-> corresponds to the window size for the calculation of long-term EMA
    The difference between these two gives the MACD line
    EMA of the MACD line gives the signal line(signal  to buy or sells)
    """
    df['MACD510'] = TA.MACD(df, period_fast=5, period_slow=10).iloc[:, 0]
    df['MACD520'] = TA.MACD(df, period_fast=5, period_slow=20).iloc[:, 0]
    df['MACD1020'] = TA.MACD(df, period_fast=10, period_slow=20).iloc[:, 0]
    df['MACD1520'] = TA.MACD(df, period_fast=15, period_slow=20).iloc[:, 0]
    df['MACD1226'] = TA.MACD(df, period_fast=12, period_slow=26).iloc[:, 0]

    # Momentum indicator
    """
    Difference between the current closing price and the closing price n periods ago
    where n is the window size
    """
    df['MOM10'] = TA.MOM(df, 10)
    df['MOM15'] = TA.MOM(df, 15)
    df['MOM20'] = TA.MOM(df, 20)

    # Rate of change
    """
    ROC= (Current closing price-closing price n periods ago)/closing price n periods ago
    """
    df['ROC5'] = TA.ROC(df, 5)
    df['ROC10'] = TA.ROC(df, 10)
    df['ROC20'] = TA.ROC(df, 20)

    # Percentage price oscillator
    """
    PPO=((short term moving average-long term moving average)/long term moving avg)*100
    Signal line-> SMA of PPO line produces the signal line
    """
    df['PPO'] = TA.PPO(df).iloc[:, 0]
    
    # Relative strength index
    """
    RSI=100-(100/(1+RS))
    here RS or relative strength is the avg of upward price changes divided by the 
    avg of downward price changes for a specified period
    """
    df['RSI14'] = TA.RSI(df, 14)
    df['RSI8'] = TA.RSI(df, 8)

    # Stochastic oscillator indicators
    """
    %K line: (Closing price-lowest low)/(Highest high-lowest low) * 100 over a specified period
    %D line: SMA of K line over a given period usually 3
    """
    df['fastk']=TA.STOCH(df)
    df['fastd']=TA.STOCHD(df)

    # Ultimate oscillator(ULTOSC)
    """
    Captures momentum across three different timeframes
    UO=[(4 * Short Average BP) + (2 * Medium Average BP) + (Long Average BP)] /
     [(4 * Short Average TR) + (2 * Medium Average TR) + (Long Average TR)]
     where BP is the buying pressure -> difference between today's close and the minimum of today's low
     or yesterday's close 
    """
    df['ULTOSC']=TA.UO(df)

    # William's %R (WILLR) 14 day period
    """
    %R= (highest high-close)/(highest high-lowest low) * -100
    """
    df['WILLR']=TA.WILLIAMS(df)

    # Average true range
    """
    ATR=SMA(TR,n)/n
    """
    df['ATR7']=TA.ATR(df,7)
    df['ATR14']=TA.ATR(df,14)

    # True range
    """
    TR= max(high-low,|high-previous close|,|low-previous close|)
    """
    df['Trange']=TA.TR(df)

    # Typical price
    """
    TP= (high+low+close)/3
    """
    df['TYPPRICE']=TA.TP(df)

    # Vortex indicators (used to identify aa new trend or the direction of an ongoing trend)
    """
    +VI= current high - previous low
    -VI= current low - previous high
    """
    df['VIn']=TA.VORTEX(df)['VIn']
    df['VIp']=TA.VORTEX(df)['VIp']

    # Money flow volume
    """
    Money flow= Typical price*volume
    """
    if 'volume' in df.columns:
        df['MFV']=TA.MFI(df)
        del(df['volume'])

    del(df['open'])
    del(df['high'])
    del(df['low'])   
    return df
    
def exponential_smooth(data,alpha):
    """
    This function performs exponential smoothing on a dataset 
    to reduce noise and make values less rigid. Exponential 
    smoothing is a technique commonly used in time series analysis 
    to assign exponentially decreasing weights to past observations.
    """
    return data.ewm(alpha=alpha).mean()

def make_prediction(data,lookahead):
    """
    Function to produce truth values
    At a given row, it moves the dataset by the window parameter
    in backward direction to see if the closing price increased (1) or decreased (0)
    """
    
    prediction=(data.shift(-lookahead)['close']>=data['close'])
    prediction=prediction.iloc[:-lookahead]
    data['pred']=prediction.astype(int)

    return data





       






