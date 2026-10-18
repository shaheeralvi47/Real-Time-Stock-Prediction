import os
import sqlite3
import time
import threading
from datetime import datetime, timedelta

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "stocks.db")

DEFAULT_TICKER = "AAPL"
CACHE_TTL = 300  # seconds
PREDICTION_DAYS = 5

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

cache = {}
cache_lock = threading.Lock()
scheduler = None
scheduler_lock = threading.Lock()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            ticker TEXT PRIMARY KEY,
            added_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            threshold REAL NOT NULL,
            triggered INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT OR IGNORE INTO settings (key, value) VALUES ('refresh_interval', '60')"
    )
    conn.commit()
    conn.close()


def get_setting(key, default=None):
    conn = get_db()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key, value):
    conn = get_db()
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()
    conn.close()


def get_cached(key):
    with cache_lock:
        entry = cache.get(key)
        if entry and time.time() - entry["time"] < CACHE_TTL:
            return entry["data"]
    return None


def set_cached(key, data):
    with cache_lock:
        cache[key] = {"time": time.time(), "data": data}


def _flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            c[0] if isinstance(c, tuple) else c for c in df.columns
        ]
    return df


def fetch_history(ticker, period="1y", interval="1d"):
    key = f"hist:{ticker.upper()}:{period}:{interval}"
    cached = get_cached(key)
    if cached is not None:
        return cached
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            return None
        df = _flatten_columns(df)
        df = df.dropna(subset=["Close"])
        if df.empty:
            return None
        set_cached(key, df)
        return df
    except Exception:
        return None


def get_latest_quote(ticker):
    df = fetch_history(ticker, period="5d", interval="1d")
    if df is None:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    price = float(last["Close"])
    prev_close = float(prev["Close"])
    change = price - prev_close
    pct = (change / prev_close * 100) if prev_close else 0.0
    volume = float(last.get("Volume", 0) or 0)
    return {
        "ticker": ticker.upper(),
        "price": round(price, 2),
        "change": round(change, 2),
        "change_pct": round(pct, 2),
        "volume": int(volume),
        "last_updated": str(df.index[-1].date()),
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def compute_indicators(df):
    df = df.copy()
    close = df["Close"]
    df["SMA20"] = close.rolling(window=20, min_periods=1).mean()
    df["SMA50"] = close.rolling(window=50, min_periods=1).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
    return df


def predict_prices(df, days=PREDICTION_DAYS):
    close = df["Close"].dropna()
    if len(close) < 15:
        return None, None
    data = pd.DataFrame({"close": close})
    for lag in range(1, 6):
        data[f"lag{lag}"] = data["close"].shift(lag)
    data = data.dropna()
    if len(data) < 10:
        return None, None

    feature_cols = [f"lag{i}" for i in range(1, 6)]
    X = data[feature_cols].to_numpy()
    y = data["close"].to_numpy()

    model = LinearRegression()
    model.fit(X, y)

    current = data[feature_cols].iloc[-1].to_numpy().tolist()
    forecasts = []
    for _ in range(days):
        pred = float(model.predict(np.array([current]))[0])
        forecasts.append(pred)
        current = [pred] + current[:-1]

    forecast_dates = []
    d = close.index[-1]
    while len(forecast_dates) < days:
        d = d + timedelta(days=1)
        if d.weekday() < 5:
            forecast_dates.append(d)

    return forecast_dates, forecasts


def build_chart(df, ticker, forecast_dates, forecasts):
    df = compute_indicators(df)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(
            f"{ticker.upper()} - Price & Forecast",
            "RSI (14)",
            "MACD (12, 26, 9)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Close",
            line=dict(color="#00d4ff", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SMA20"],
            name="SMA 20",
            line=dict(color="#ff9f43", width=1.5, dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SMA50"],
            name="SMA 50",
            line=dict(color="#70a1ff", width=1.5, dash="dot"),
        ),
        row=1,
        col=1,
    )

    if forecast_dates and forecasts:
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecasts,
                name="Predicted (experimental)",
                line=dict(color="#bb6bd9", width=2, dash="dash"),
                mode="lines+markers",
                marker=dict(symbol="diamond", size=7),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            name="RSI",
            line=dict(color="#00d4ff", width=1.5),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dot", line_color="#ff4757", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#2ed573", row=2, col=1)

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD"],
            name="MACD",
            line=dict(color="#00d4ff", width=1.5),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD_SIGNAL"],
            name="Signal",
            line=dict(color="#ff4757", width=1.5),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["MACD_HIST"],
            name="Histogram",
            marker_color=np.where(df["MACD_HIST"] >= 0, "#2ed573", "#ff4757"),
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=760,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40, l=60, r=40),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


@app.route("/")
def index():
    ticker = (request.args.get("ticker") or DEFAULT_TICKER).strip().upper()
    quote = get_latest_quote(ticker)
    chart_html = None
    indicators = None
    forecast_rows = None
    error = None

    if quote is None:
        error = f"Could not retrieve data for '{ticker}'. Check the ticker symbol and your network connection."
    else:
        df = fetch_history(ticker, period="1y", interval="1d")
        if df is None:
            error = f"Could not load history for '{ticker}'."
        else:
            df = compute_indicators(df)
            chart_html = build_chart(df, ticker, *predict_prices(df))
            last = df.iloc[-1]
            indicators = {
                "sma20": round(float(last["SMA20"]), 2) if pd.notna(last["SMA20"]) else None,
                "sma50": round(float(last["SMA50"]), 2) if pd.notna(last["SMA50"]) else None,
                "rsi": round(float(last["RSI"]), 1) if pd.notna(last["RSI"]) else None,
                "macd": round(float(last["MACD"]), 4) if pd.notna(last["MACD"]) else None,
                "macd_signal": round(float(last["MACD_SIGNAL"]), 4)
                if pd.notna(last["MACD_SIGNAL"])
                else None,
            }
            dates, preds = predict_prices(df)
            if dates and preds:
                forecast_rows = [(d.strftime("%Y-%m-%d"), round(p, 2)) for d, p in zip(dates, preds)]

    conn = get_db()
    watchlist = [
        {"ticker": r["ticker"], "added_at": r["added_at"]}
        for r in conn.execute(
            "SELECT * FROM watchlist ORDER BY added_at DESC"
        ).fetchall()
    ]
    alerts = [
        dict(r)
        for r in conn.execute("SELECT * FROM alerts ORDER BY created_at DESC").fetchall()
    ]
    conn.close()

    watchlist_quotes = []
    for w in watchlist:
        q = get_latest_quote(w["ticker"])
        if q:
            watchlist_quotes.append(q)

    triggered_alerts = [a for a in alerts if a.get("triggered")]

    refresh_interval = int(get_setting("refresh_interval", 60))

    return render_template(
        "index.html",
        ticker=ticker,
        quote=quote,
        error=error,
        chart_html=chart_html,
        indicators=indicators,
        forecast_rows=forecast_rows,
        watchlist=watchlist_quotes,
        alerts=alerts,
        triggered_alerts=triggered_alerts,
        refresh_interval=refresh_interval,
    )


@app.route("/api/quote/<ticker>")
def api_quote(ticker):
    q = get_latest_quote(ticker)
    if q is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(q)


@app.route("/watchlist/add", methods=["POST"])
def watchlist_add():
    ticker = (request.form.get("ticker") or "").strip().upper()
    if not ticker:
        flash("Enter a ticker symbol.", "error")
        return redirect(url_for("index"))
    q = get_latest_quote(ticker)
    if q is None:
        flash(f"Invalid ticker '{ticker}'.", "error")
        return redirect(url_for("index"))
    conn = get_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist (ticker, added_at) VALUES (?, ?)",
            (ticker, datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
        flash(f"{ticker} added to watchlist.", "success")
    except sqlite3.Error:
        flash("Database error adding watchlist item.", "error")
    finally:
        conn.close()
    return redirect(url_for("index"))


@app.route("/watchlist/remove/<ticker>", methods=["POST"])
def watchlist_remove(ticker):
    conn = get_db()
    conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))
    conn.commit()
    conn.close()
    flash(f"{ticker.upper()} removed from watchlist.", "success")
    return redirect(url_for("index"))


@app.route("/alerts/add", methods=["POST"])
def alerts_add():
    ticker = (request.form.get("ticker") or "").strip().upper()
    alert_type = request.form.get("alert_type")
    threshold = request.form.get("threshold", type=float)
    if not ticker or not alert_type or threshold is None:
        flash("Complete all alert fields.", "error")
        return redirect(url_for("index"))
    if alert_type not in ("above", "below", "change_pct"):
        flash("Invalid alert type.", "error")
        return redirect(url_for("index"))
    if get_latest_quote(ticker) is None:
        flash(f"Invalid ticker '{ticker}'.", "error")
        return redirect(url_for("index"))
    conn = get_db()
    conn.execute(
        "INSERT INTO alerts (ticker, alert_type, threshold, triggered, created_at) "
        "VALUES (?, ?, ?, 0, ?)",
        (ticker, alert_type, threshold, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()
    flash(f"Alert added for {ticker}.", "success")
    return redirect(url_for("index"))


@app.route("/alerts/remove/<int:alert_id>", methods=["POST"])
def alerts_remove(alert_id):
    conn = get_db()
    conn.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
    conn.commit()
    conn.close()
    flash("Alert removed.", "success")
    return redirect(url_for("index"))


@app.route("/settings/refresh", methods=["POST"])
def settings_refresh():
    interval = request.form.get("refresh_interval", type=int)
    if not interval or interval < 15:
        flash("Interval must be at least 15 seconds.", "error")
    else:
        set_setting("refresh_interval", str(interval))
        flash(f"Refresh interval set to {interval}s.", "success")
    return redirect(url_for("index"))


def check_alerts():
    with app.app_context():
        conn = get_db()
        rows = conn.execute(
            "SELECT * FROM alerts WHERE triggered = 0"
        ).fetchall()
        for row in rows:
            ticker = row["ticker"]
            q = get_latest_quote(ticker)
            if q is None:
                continue
            price = q["price"]
            triggered = False
            if row["alert_type"] == "above" and price >= row["threshold"]:
                triggered = True
            elif row["alert_type"] == "below" and price <= row["threshold"]:
                triggered = True
            elif row["alert_type"] == "change_pct":
                if q["change_pct"] >= row["threshold"]:
                    triggered = True
            if triggered:
                conn.execute(
                    "UPDATE alerts SET triggered = 1 WHERE id = ?", (row["id"],)
                )
                app.logger.info(
                    "ALERT TRIGGERED: %s %s >= %s (price %s, change %.2f%%)",
                    ticker,
                    row["alert_type"],
                    row["threshold"],
                    price,
                    q["change_pct"],
                )
        conn.commit()
        conn.close()


def start_scheduler():
    global scheduler
    with scheduler_lock:
        if scheduler is not None:
            return scheduler
        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(
            func=check_alerts,
            trigger=IntervalTrigger(seconds=60),
            id="alert_check",
            replace_existing=True,
        )
        scheduler.start()
        app.logger.info("Scheduler started")
    return scheduler


init_db()
start_scheduler()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
