from flask import Flask, render_template, jsonify
import pandas as pd
import matplotlib.pyplot as plt
from src.ml_trade.features import get_indicator_data, exponential_smooth, make_prediction
from src.ml_trade.train import validation
from src.ml_trade.test import returns, calculate_max_drawdown, calculate_sharpe_ratio, calculate_win_loss_ratio, calculate_calmar_ratio
import numpy as np

app = Flask(__name__)
def load_and_preprocess_data():
    df_main = pd.read_csv('data/reversed_NIFTY_50_23_years.csv', usecols=['open', 'high', 'low', 'close'])
    df_main = get_indicator_data(df_main)
    df_main.dropna(inplace=True)
    return df_main

@app.route('/')
def index():
    return render_template('index.html')

# Route to get preprocessed data
@app.route('/get_data')
def get_data():
    df_main = load_and_preprocess_data()
    return df_main.to_json(orient='records')

# Function to generate PnL plot
def generate_pnl_plot_1(df_test, model):
    plt.figure(figsize=(10, 6))
    df_test['pnl'].plot()
    plt.xlabel('Time')
    plt.ylabel('PnL')
    plt.title('Profit and Loss (PnL) Plot')
    plt.grid(True)
    plt.savefig('static/pnl_plot_1.png')  # Save the plot as an image
    plt.close()

def generate_pnl_plot_2(df_test, model):
    plt.figure(figsize=(10, 6))
    df_test['pnl'].plot()
    plt.xlabel('Time')
    plt.ylabel('PnL')
    plt.title('Profit and Loss (PnL) Plot')
    plt.grid(True)
    plt.savefig('static/pnl_plot_2.png')  # Save the plot as an image
    plt.close()


# Route to train models, generate plots, and return evaluation metrics
@app.route('/train_and_evaluate')
def train_and_evaluate():
    df_main = load_and_preprocess_data()
    period = 1000  # number of days to test on => 1000 means roughly 4 years
    df_train = df_main[:-period].reset_index(drop=True)
    df_test = df_main[-period:].reset_index(drop=True)

    data = exponential_smooth(df_train, 0.8)
    data = make_prediction(data, lookahead=10)
    data.dropna(inplace=True)

    modelRF, modelXT = validation(data)

    cumulative_return_RF = returns(modelRF, df_test, 100000, window=5, volume=40, confidence=0.5, exponent=2.4, alpha=0.5)*100
    max_drawdown_RF = calculate_max_drawdown(df_test['pnl'])*100
    sharpe_ratio_RF = calculate_sharpe_ratio(df_test['pnl'], 0.15)  # 15% risk-free rate
    calmar_ratio_RF = calculate_calmar_ratio(df_test['pnl'])
    win_loss_ratio_RF = calculate_win_loss_ratio(df_test['pnl'])
    generate_pnl_plot_1(df_test, modelRF)

    cumulative_return_XT = returns(modelXT, df_test, 100000, window=5, volume=40, confidence=0.5, exponent=2.4, alpha=0.5)*100
    max_drawdown_XT = calculate_max_drawdown(df_test['pnl'])*100
    sharpe_ratio_XT = calculate_sharpe_ratio(df_test['pnl'], 0.15)
    calmar_ratio_XT = calculate_calmar_ratio(df_test['pnl'])
    win_loss_ratio_XT = calculate_win_loss_ratio(df_test['pnl'])
    generate_pnl_plot_2(df_test, modelXT) 

    return render_template('results.html',
                           cumulative_return_RF=cumulative_return_RF,
                           cumulative_return_XT=cumulative_return_XT,
                           max_drawdown_RF=max_drawdown_RF,
                           max_drawdown_XT=max_drawdown_XT,
                           sharpe_ratio_RF=sharpe_ratio_RF,
                           sharpe_ratio_XT=sharpe_ratio_XT,
                           calmar_ratio_RF=calmar_ratio_RF,
                           calmar_ratio_XT=calmar_ratio_XT,
                           win_loss_ratio_RF=win_loss_ratio_RF,
                           win_loss_ratio_XT=win_loss_ratio_XT)

@app.route('/graphs')
def show_results():
    return render_template('graphs.html')

if __name__ == '__main__':
    app.run(debug=True)


