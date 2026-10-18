# Stock Prediction & Market Monitoring Dashboard

Lightweight Flask dashboard for stock prices, technical indicators, scikit-learn price forecasts, watchlists, and alerts.

## Prerequisites

- Python 3.9+
- Internet access (Yahoo Finance via `yfinance`; no API key required)

## Install and run (Windows PowerShell)

```powershell
# 1. Create the project folder and enter it
cd D:\projects
mkdir stock-dashboard
cd stock-dashboard

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the application
python app.py
```

Open your browser at **http://127.0.0.1:5000**.

## Features

- Dashboard KPI cards: latest price, daily change, percentage change, volume
- Interactive Plotly chart: price, SMA 20/50, RSI, MACD, and a 5-session Linear Regression forecast (flagged experimental)
- Stock search by ticker (AAPL, TSLA, MSFT, NVDA, etc.)
- Technical indicators computed with Pandas: SMA 20, SMA 50, RSI, MACD
- Market alerts: price above/below and percentage-change, monitored in the background with APScheduler; triggered alerts shown on the dashboard (no duplicates)
- Watchlist: add/remove tickers with latest prices stored in SQLite
- Auto-refresh with configurable interval and last-updated timestamp; data cached to limit Yahoo Finance requests
- SQLite persistence for watchlist and alerts (survives restarts)

## Notes

- Yahoo Finance data is the latest available and may be delayed; it is not guaranteed real-time.
- Forecasts are simple Linear Regression estimates on lagged features — not financial advice.
- No authentication or external API keys are required.
