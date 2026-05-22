import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. Load the trained model (and label mappings)
# ============================================================
@st.cache_resource
def load_model():
    data = joblib.load("xgboost_3class_1m.pkl")
    return data["model"], data["label_map"], data["inv_map"]

model, label_map, inv_map = load_model()

# ============================================================
# 2. Fetch latest 1‑minute data for the 10 tickers
# ============================================================
TICKERS = ["AAPL","NVDA","TSLA","MSFT","AMZN","AMD","GOOG","AVGO","INTC","MU"]
INTERVAL = "1m"
LOOKBACK_DAYS = 5   # enough to compute all indicators (max window 20 bars)

@st.cache_data(ttl=60)   # cache for 60 seconds
def fetch_data(tickers):
    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_DAYS)
    frames = []
    for t in tickers:
        df = yf.download(t, interval=INTERVAL, start=start, end=end,
                         progress=False, auto_adjust=False, prepost=False)
        if df.empty:
            continue
        df.index = df.index.tz_localize(None)
        # flatten MultiIndex columns if any
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df[['Open','High','Low','Close','Volume']].copy()
        df['Ticker'] = t
        df.reset_index(inplace=True)
        df.rename(columns={'index':'Datetime'}, inplace=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    full['Datetime'] = pd.to_datetime(full['Datetime'])
    full.sort_values(['Ticker','Datetime'], inplace=True)
    # Ensure numeric prices
    full[['Open','High','Low','Close','Volume']] = full[['Open','High','Low','Close','Volume']].astype(float)
    return full

# ============================================================
# 3. Feature engineering (exactly as in training)
# ============================================================
def create_features_1m(group):
    group = group.copy().reset_index(drop=True)

    group["ret1"] = group["Close"].pct_change(1)
    group["ret3"] = group["Close"].pct_change(3)
    group["ret5"] = group["Close"].pct_change(5)
    group["log_ret"] = np.log(group["Close"] / group["Close"].shift(1))

    for p in [5, 9, 12, 20]:
        group[f"ema_{p}"] = ta.trend.ema_indicator(group["Close"], window=p)
        group[f"ema_dist_{p}"] = (group["Close"] - group[f"ema_{p}"]) / group[f"ema_{p}"]
        group[f"ema_slope_{p}"] = group[f"ema_{p}"].diff()
    group["ema_spread_5_20"] = group["ema_5"] - group["ema_20"]

    group["rsi_14"] = ta.momentum.rsi(group["Close"], window=14)
    group["rsi_7"]  = ta.momentum.rsi(group["Close"], window=7)
    group["rsi_diff"] = group["rsi_14"].diff()

    macd = ta.trend.MACD(group["Close"])
    group["macd"] = macd.macd()
    group["macd_signal"] = macd.macd_signal()
    group["macd_hist"] = macd.macd_diff()

    group["atr"] = ta.volatility.average_true_range(
        high=group["High"], low=group["Low"], close=group["Close"], window=14)
    group["atr_pct"] = group["atr"] / group["Close"]

    bb = ta.volatility.BollingerBands(close=group["Close"], window=20, window_dev=2)
    group["bb_h"] = bb.bollinger_hband()
    group["bb_l"] = bb.bollinger_lband()
    group["bb_m"] = bb.bollinger_mavg()
    group["bb_width"] = (group["bb_h"] - group["bb_l"]) / group["bb_m"]
    group["bb_pos"] = (group["Close"] - group["bb_l"]) / (group["bb_h"] - group["bb_l"])

    group["date"] = group["Datetime"].dt.date
    tp = (group["High"] + group["Low"] + group["Close"]) / 3
    cum_vp  = (tp * group["Volume"]).groupby(group["date"]).cumsum()
    cum_vol = group["Volume"].groupby(group["date"]).cumsum()
    group["vwap"] = cum_vp / cum_vol
    group["vwap_dist"] = (group["Close"] - group["vwap"]) / group["vwap"]
    group.drop(columns=["date"], inplace=True)

    group["vol_sma20"] = group["Volume"].rolling(20, min_periods=1).mean()
    group["vol_ratio"] = group["Volume"] / group["vol_sma20"]
    group["vol_change"] = group["Volume"].pct_change()

    group["body"] = abs(group["Close"] - group["Open"])
    group["upper_wick"] = group["High"] - np.maximum(group["Open"], group["Close"])
    group["lower_wick"] = np.minimum(group["Open"], group["Close"]) - group["Low"]
    group["range"] = group["High"] - group["Low"]
    group["body_ratio"] = group["body"] / (group["range"] + 1e-9)

    group["mom3"] = group["Close"] - group["Close"].shift(3)
    group["mom5"] = group["Close"] - group["Close"].shift(5)
    group["roc5"] = ta.momentum.roc(group["Close"], window=5)

    stoch = ta.momentum.StochRSIIndicator(close=group["Close"], window=14)
    group["stoch_rsi"] = stoch.stochrsi()

    group["above_ema20"] = (group["Close"] > group["ema_20"]).astype(int)
    group["above_vwap"]  = (group["Close"] > group["vwap"]).astype(int)
    group["bull_candle"] = (group["Close"] > group["Open"]).astype(int)

    group["std10"] = group["Close"].rolling(10, min_periods=1).std()
    group["std20"] = group["Close"].rolling(20, min_periods=1).std()
    group["vol_ratio"] = group["std10"] / group["std20"]

    for lag in [1,2,3,5]:
        group[f"close_lag{lag}"] = group["Close"].shift(lag)
        group[f"vol_lag{lag}"]   = group["Volume"].shift(lag)
        group[f"rsi_lag{lag}"]   = group["rsi_14"].shift(lag)

    group["hour"]   = group["Datetime"].dt.hour
    group["minute"] = group["Datetime"].dt.minute
    group["dow"]    = group["Datetime"].dt.dayofweek
    return group

# ============================================================
# 4. Predict and find strongest breakout candidates
# ============================================================
def get_breakout_candidates(df, margin=0.1):
    """
    Returns a DataFrame with per-ticker predictions and a flag
    indicating a valid breakout signal (bull/bear).
    """
    # Build features per ticker
    feature_list = []
    for ticker, grp in df.groupby("Ticker"):
        feat = create_features_1m(grp)
        feature_list.append(feat)
    if not feature_list:
        return None
    feature_df = pd.concat(feature_list, ignore_index=True)

    # Market feature
    mkt = feature_df.groupby('Datetime')['log_ret'].mean().rename('mkt_ret').reset_index()
    feature_df = feature_df.merge(mkt, on='Datetime', how='left')
    feature_df['rel_str'] = feature_df['log_ret'] - feature_df['mkt_ret']

    # Clean NaNs
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_df.dropna(inplace=True)
    if feature_df.empty:
        return None

    # Features for prediction (exclude non-feature columns)
    feat_cols = [c for c in feature_df.columns if c not in ("Datetime","Ticker","target")]
    X = feature_df[feat_cols]

    # Predict probabilities (output shape: (n,3) with columns: 0=Down, 1=Neutral, 2=Up)
    proba = model.predict_proba(X)
    prob_down = proba[:, 0]   # mapped label 0 -> original -1 (Down)
    prob_neutral = proba[:, 1] # mapped label 1 -> original 0
    prob_up = proba[:, 2]      # mapped label 2 -> original 1 (Up)

    feature_df["prob_up"] = prob_up
    feature_df["prob_down"] = prob_down
    feature_df["prob_neutral"] = prob_neutral

    # Determine breakout signal using margin rule:
    #   Long  if P(Up) > P(Neutral) + margin
    #   Short if P(Down) > P(Neutral) + margin
    feature_df["signal"] = 0
    feature_df.loc[feature_df["prob_up"] > feature_df["prob_neutral"] + margin, "signal"] = 1
    feature_df.loc[feature_df["prob_down"] > feature_df["prob_neutral"] + margin, "signal"] = -1

    # The "strength" of the signal = max(P(Up), P(Down)) - P(Neutral)
    feature_df["strength"] = np.maximum(feature_df["prob_up"], feature_df["prob_down"]) - feature_df["prob_neutral"]

    # Get the most recent bar for each ticker
    last_rows = feature_df.groupby("Ticker").last().reset_index()
    return last_rows[["Ticker","prob_up","prob_down","prob_neutral","signal","strength"]]

# ============================================================
# 5. Streamlit UI
# ============================================================
st.set_page_config(page_title="Breakout Scanner (3‑Class)", layout="wide")
st.title("📈 5‑Minute Breakout Scanner (Up / Down)")
st.markdown("""
Uses a trained XGBoost model to predict **0.1% moves in 5 minutes**.  
Signals are filtered using a **margin rule**:
- **Bull** if P(Up) > P(Neutral) + margin
- **Bear** if P(Down) > P(Neutral) + margin
""")

col1, col2 = st.columns(2)
margin = col1.slider("Margin (strictness)", 0.0, 0.3, 0.1, 0.01)
auto_refresh = col2.checkbox("Auto‑refresh every 60 seconds")

if auto_refresh:
    st.rerun()   # auto-rerun triggers the whole script again

if st.button("Scan Now"):
    with st.spinner("Fetching latest 1‑minute data and computing predictions..."):
        raw_data = fetch_data(TICKERS)
        if raw_data.empty:
            st.error("No data fetched. Check your internet or ticker symbols.")
        else:
            results = get_breakout_candidates(raw_data, margin=margin)
            if results is None or results.empty:
                st.warning("Not enough recent data to compute features. Try again in a few minutes.")
            else:
                # Filter only tickers with a signal
                signals = results[results["signal"] != 0].copy()
                if signals.empty:
                    st.info("No ticker meets the breakout criteria with the current margin.")
                else:
                    # Sort by strength (descending)
                    signals = signals.sort_values("strength", ascending=False)
                    strongest = signals.iloc[0]
                    st.success(f"🔝 **{strongest['Ticker']}** – strongest signal!")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Signal", "Bullish 📈" if strongest["signal"]==1 else "Bearish 📉")
                    c2.metric("Strength", f"{strongest['strength']:.2%}")
                    c3.metric("Prob(Up) / Prob(Down)", f"{strongest['prob_up']:.2%} / {strongest['prob_down']:.2%}")

                # Show full table of all tickers (with signals on top)
                st.subheader("All Tickers (sorted by breakout potential)")
                display_df = results.copy()
                display_df["Direction"] = display_df["signal"].map({1:"Bull", -1:"Bear", 0:"None"})
                display_df["Strength"] = display_df["strength"].apply(lambda x: f"{x:.2%}")
                display_df = display_df.sort_values("strength", ascending=False)
                st.dataframe(
                    display_df[["Ticker","Direction","Strength","prob_up","prob_down","prob_neutral"]],
                    use_container_width=True
                )
