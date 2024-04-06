
from sklearn.preprocessing import StandardScaler
import math
import numpy as np

def returns(model, data, amount, window=5, volume=40, confidence=0.5, exponent=2.4, alpha=0.5):
    """
    Function to execute the strategy and compute the returns.

    Args:
        model (sklearn model): The model to be used for predictions.
        data (pd.DataFrame): The dataframe containing the data.
        amount (float): The amount of money to be used for trading.
        window (int, optional): The window size to be used for predictions. Set default value to 5.
        volume (int, optional): The maximum volume of shares that are allowed to be traded in a single day.
        conf (float, optional): The initial confidence multiplier that decides the volume to trade.
        exp (float, optional): The exponent to be used for the confidence multiplier. 
        A higher value tends to bring a lower risk, but the overall returns might reduce.
        alpha (float, optional): The moving average coefficient for confidence.

    Returns:
        float: The returns as a decimal.
    """

    i=0 # Initialise the iterator to 0
    holding=0 # Initial number of shares held
    balance=amount 
    scaler=StandardScaler()
    prediction=np.zeros(window) # Stores initial predictions, this is used to calculate confidence
    data["pnl"]=amount # Column to store the amount of profit or loss
    while True:
        df_train=data.iloc[i*window:i*window+window]
        if len(df_train)< window:
            break
        features=[x for x in df_train.columns if x not in ["pred","pnl"]]
        X=df_train[features]
        X=scaler.fit_transform(X)
        new_prediction=model.predict(X)
        for j in range(window):
            current_rate=df_train["close"].values[j]
            if i>=1:
               prev_rate=data.iloc[(i-1)*window+j]["close"]
               result=prediction[j]==(current_rate>prev_rate) # 1 if previous prediction was correct, 0 if wrong
               confidence=(1-alpha)*confidence+alpha*result
            shares=math.floor(volume * confidence**exponent)
            max_shares=math.floor(balance/current_rate)
            if new_prediction[j] == 1:
                temp = min(shares, max_shares) # number of shares to buy
                balance -= temp * current_rate # reduce balance
                holding += temp # increase holding
            else:
                temp = min(shares, holding) # number of shares to sell
                holding -= temp # reduce holding
                balance += temp * current_rate # increase balance
                       
            data.iloc[i * window + j, data.columns.get_loc("pnl")] = (
                balance + holding * current_rate # update profit/loss column
            )
        prediction = new_prediction # update prediction
        i += 1
    closing_rate = data["close"].values[-1]
    ret = (balance + holding * closing_rate) / amount
    return ret

def calculate_wins_losses(pnl):
    wins = 0
    losses = 0
    prev_value = pnl.iloc[0]  # Initialize with the first value
    
    for value in pnl[1:]:  # Start from the second value
        if value > prev_value:
            wins += 1
        elif value < prev_value:
            losses += 1
        
        prev_value = value  # Update the previous value for the next iteration
    
    return wins, losses


def calculate_win_loss_ratio(pnl):
    """
    Function to calculate the win loss ratio from a dataframe column 'pnl'.

    Args:
        pnl (pd.Series): A pandas series representing the profit/loss values.

    Returns:
        float: The win loss ratio.
    """
    wins,losses = calculate_wins_losses(pnl)  
    win_loss_ratio = wins / losses if losses != 0 else np.inf  # Calculate profit factor
    return win_loss_ratio

def calculate_max_drawdown(pnl):
    """
    Function to calculate the maximum drawdown from a dataframe column 'pnl'.

    Args:
        pnl (pd.Series): A pandas series representing the profit/loss values.

    Returns:
        float: The maximum drawdown value as a decimal.
    """
    cummax = pnl.cummax()  # Calculate cumulative maximum
    drawdown = (
        pnl - cummax
    ) / cummax  # Calculate drawdown as a percentage of the cumulative maximum
    max_drawdown = drawdown.min()  # Get the minimum drawdown value
    return max_drawdown

def calculate_sharpe_ratio(pnl, risk_free_rate):
    """
    Function to calculate the Sharpe ratio from a dataframe column 'pnl' and a risk-free rate.

    Args:
        pnl (pd.Series): A pandas series representing the profit/loss values.
        risk_free_rate (float): The risk-free rate for the given period.

    Returns:
        float: The Sharpe ratio.
    """
    returns = pnl.pct_change()  # Calculate the returns as the percentage change of pnl
    annualized_returns = (
        np.mean(returns) * 252
    )  # Calculate the annualized returns assuming 252 trading days in a year
    volatility = np.std(returns) * np.sqrt(
        252
    )  # Calculate the volatility (standard deviation of returns)
    sharpe_ratio = (
        annualized_returns - risk_free_rate
    ) / volatility  # Calculate the Sharpe ratio
    return sharpe_ratio

def calculate_calmar_ratio(pnl):
    """
    Function to calculate the Calmar ratio.

    Args:
        pnl (pd.Series): A pandas series representing the profit/loss values.    

    Returns:
        float: The Calmar ratio.
    """
    max_drawdown=calculate_max_drawdown(pnl)
    returns = pnl.pct_change()  # Calculate the returns as the percentage change of pnl
    annualized_returns = (
        np.mean(returns) * 252
    )  # Calculate the annualized returns assuming 252 trading days in a year
    if max_drawdown == 0:  # Avoid division by zero
        return float('inf') if annualized_returns > 0 else 0.0
    
    return annualized_returns / max_drawdown
    

            












    






