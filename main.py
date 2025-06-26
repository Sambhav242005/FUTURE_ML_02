import numpy as np
import pandas as pd
import gradio as gr
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from data import load_sales_dataset

# 1. Create Lag Features
def create_lag_features(series, lag=5):
    X, y = [], []
    for i in range(lag, len(series)):
        X.append(series[i - lag:i].flatten())
        y.append(series[i].flatten())
    return np.array(X), np.array(y)

# 2. Load Stock Metadata
stock_df = load_sales_dataset(filename="symbols_valid_meta.csv")
stock_options = stock_df[["Security Name", "Symbol"]].drop_duplicates()
stock_choices = stock_options["Security Name"].tolist()

# 3. Main Function with Stock Selection
def stock_prediction(stock_choice):
    
    # Get the corresponding symbol for the selected Security Name
    symbol_row = stock_options[stock_options["Security Name"] == stock_choice]
    
    if symbol_row.empty:
        return "Error: Invalid stock selection.", None
    
    symbol = symbol_row.iloc[0]["Symbol"]

    # Load Data
    df = load_sales_dataset(filename=f"stocks/{symbol}.csv")
    df['Close'] = df['Close'].bfill()
    data = df[['Close']].values

    # Create Features
    lag = 5
    X, y = create_lag_features(data, lag)

    # Train/Test Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False,
    )

    # Scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict & Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    # Convert to percentage based on mean price
    mean_price = np.mean(y)

    train_rmse_perc = (train_rmse / mean_price) * 100
    test_rmse_perc = (test_rmse / mean_price) * 100
    train_mae_perc = (train_mae / mean_price) * 100
    test_mae_perc = (test_mae / mean_price) * 100

    metrics_text = (
        f"### Evaluation Metrics\n"
        f"- **Train RMSE:** {train_rmse_perc:.2f}%\n"
        f"- **Train MAE:** {train_mae_perc:.2f}%\n"
        f"- **Test RMSE:** {test_rmse_perc:.2f}%\n"
        f"- **Test MAE:** {test_mae_perc:.2f}%"
    )



    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(y)), y, label='Actual', color='black',linewidth=4)
    ax.plot(range(lag, lag + len(train_pred)), train_pred, label='Train Pred', color='orange',linewidth=1)
    ax.plot(range(lag + len(train_pred), lag + len(train_pred) + len(test_pred)), test_pred, label='Test Pred', color='green',linewidth=1)
    ax.set_title(f'Stock Price Prediction ({stock_choice}, Linear Regression, Lag 5)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Stock Price')
    ax.legend()
    ax.grid()

    return metrics_text, fig

# 4. Gradio UI
with gr.Blocks(title="Stock Price Prediction - Linear Regression (Lag 5)") as interface:
    
    gr.Markdown("## Stock Price Prediction - Linear Regression (Lag 5)")
    gr.Markdown("Select a stock to predict its price using a Linear Regression model with lag=5.")
    
    stock_dropdown = gr.Dropdown(choices=stock_choices, label="Select Stock")
    predict_btn = gr.Button("Predict")

    metrics_output = gr.Markdown(label="Evaluation Metrics")
    plot_output = gr.Plot(label="Prediction Plot")

    def wrapped_prediction(stock_choice):
        return stock_prediction(stock_choice)
    
    predict_btn.click(fn=wrapped_prediction, inputs=stock_dropdown, outputs=[metrics_output, plot_output])

interface.launch()
