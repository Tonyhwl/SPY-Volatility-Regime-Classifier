import numpy as np, pandas as pd # for mathematical calculations and time-series tables
import yfinance as yf #yahoo finance to fetch historical data
import matplotlib.pyplot as plt

TICKER = 'SPY' #SPY
start_date = '2005-01-01' #20 years of data

# yf.download() returns dataframe indexed by trading dates
# we only want the close column from 'Open, High, Low, Close, Adj Close, Volume'
# realised volatility is calculated from daily close-to-close returns
price_data = yf.download(
    TICKER,
    start=start_date,
    auto_adjust = True, # 'auto_adjust = True' adjusts the prices for splits and dividends
    progress = False)[['Close']] # 'progress = False' hides the tqdm progress bar (noisy)

# yahoo has holes with empty data (holidays, partial days etc.)
#'px.dropna()' removes any rows with empty data
price_data = price_data.dropna()

daily_returns = price_data['Close'].pct_change() # computes simple daily returns: (today's closing price - yesterday's closing price) / yesterday's closing price

def compute_realised_volatility(
    daily_returns: pd.Series, #
    window_days: int = 30, # rolling window length in trading days in a month
    trading_days_per_year: int = 252) -> pd.Series: #252 trading days for ETFs

    # .rolling(window_days) builds a moving window of the last 30 days for each date
    # .std(ddof=0) computes the population standard deviation of the last 30 days for each date. 'ddof=1' is sample stdev
    rolling_std = daily_returns.rolling(window_days).std(ddof=0)

    # converting volatility measured over a day into volatility measured over a year using square root of time rule
    # this rule assumes that the returns are independent and identically distributed (i.i.d)
    realised_vol_annualised = rolling_std * np.sqrt(trading_days_per_year)

    return realised_vol_annualised

# convert continuous volatility numbers -> discrete volatility regimes such as 0 = low, 1 = mid, 2 = high
def label_volatility_regime(
    realised_vol_series: pd.Series, # annualised realised vol
    low_enter: float = 0.12, low_exit: float = 0.14,   # 12% annualised vol, values below this are labeled low (0)
    high_enter: float = 0.28, high_exit: float = 0.24  # 28% annualised vol, values >= this, are labeled high (2), anything in-between exits are mid (1)
) -> pd.Series:
    out = []
    state = np.nan  # unknown until first valid value

    for v in realised_vol_series:
        if np.isnan(v):
            out.append(np.nan)
            continue

        # initialise state on first valid point
        if np.isnan(state):
            state = 0 if v <= low_enter else (2 if v >= high_enter else 1)
            out.append(state)
            continue

        if state == 0:  # currently low
            if v >= low_exit:  # only leave low if it heats up enough
                state = 1
        elif state == 2:  # currently high
            if v <= high_exit:  # only leave high if it cools enough
                state = 1
        else:  # currently MID
            if v <= low_enter:
                state = 0
            elif v >= high_enter:
                state = 2

        out.append(state)

    return pd.Series(out, index=realised_vol_series.index)

# hysteresis prevents regime flipping between low and mid etc., regime only changes if it remains the same for a few days
def apply_hysteresis(
    regime_series: pd.Series, # pd.Series of {0, 1, 2} or NaN (raw regime labels per day)
    k_consecutive: int = 7, # how many consecutive days are in the new regime are required before flipping
) -> pd.Series:
    smoothed_labels = []  # build the output, one label per day
    current_regime = np.nan  # the regime we are in
    run_regime = None  # the candidate regime we are seeing now
    run_length = 0  # how many consecutive days we've seen run_regime

    # iterate through the raw labels in time order
    for raw_label in regime_series:
        # if today's raw label is missing, push NaN and continue
        if np.isnan(raw_label):
            smoothed_labels.append(np.nan)
            continue

        # if we haven't set an official current regime yet, take the first valid one
        if np.isnan(current_regime):
            current_regime = raw_label
            run_regime = raw_label
            run_length = 1
            smoothed_labels.append(current_regime)
            continue

        # today's raw label matches our current official regime -> no switch
        if raw_label == current_regime:
            # reset streak to this same regime (keeps logic simple)
            run_regime = raw_label
            run_length = 1
            smoothed_labels.append(current_regime)

        # today's raw label is different -> possible new regime forming
        else:
            # extend if same
            if run_regime == raw_label:
                run_length += 1
            else:
                # new label, start from 1
                run_regime = raw_label
                run_length = 1

            # switch if we've seen k_consecutive in a row
            if run_length >= k_consecutive:
                current_regime = raw_label
                # treat today as day 1 of the new official regime
                run_length = 1

            # append the official regime for today
            smoothed_labels.append(current_regime)

    # return as a Series with the same index as input
    return pd.Series(smoothed_labels, index=regime_series.index)

# organise into a neat table
analysis_df = pd.DataFrame(index=price_data.index)
analysis_df["close"] = price_data["Close"]
analysis_df["daily_returns"] = daily_returns
analysis_df["realised_vol_30d"] = compute_realised_volatility(daily_returns, window_days=30, trading_days_per_year=252)
analysis_df["regime_raw"] = label_volatility_regime(
    analysis_df["realised_vol_30d"],
    low_enter=0.12, low_exit=0.14,
    high_enter=0.28,  high_exit=0.24
)
analysis_df["regime"] = apply_hysteresis(analysis_df["regime_raw"], k_consecutive=7)
# show the last few rows to compare raw vs smoothed
print(analysis_df[["realised_vol_30d", "regime_raw", "regime"]].tail(10))

# count how many days in each (smoothed) regime
print("Counts by regime (smoothed):")
print(analysis_df["regime"].value_counts(dropna=True).sort_index())


# print today's final regime in words
regime_mapping = {0: "LOW", 1: "MID", 2: "HIGH"}
regime_today_int = int(analysis_df["regime"].dropna().iloc[-1])
vol_today = analysis_df["realised_vol_30d"].iloc[-1]
print(f"Today's regime: {regime_mapping[regime_today_int]} (30d vol = {vol_today:.2%})")


# .tail(n) prints the last n days, .head(n) prints the first n days
#print(analysis_df[["close", "daily_returns", "realised_vol_30d"]].tail(30))
#print(price_data.head(10))
#print(daily_returns.head(10))

# ------------------ regime metrics ------------------
def sharpe_daily(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0 or r.std(ddof=0) == 0:
        return np.nan
    return (r.mean() / r.std(ddof=0)) * np.sqrt(252)

def max_drawdown_from_returns(returns: pd.Series) -> float:
    r = returns.dropna()
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    return dd.min()  # negative

regime_stats = (
    analysis_df
    .assign(regime_name=analysis_df["regime"].map({0: "LOW", 1: "MID", 2: "HIGH"}))
    .groupby("regime_name")
    .agg(
        days=("regime_name", "count"),
        pct_days=("regime_name", lambda x: 100 * len(x) / len(analysis_df)),
        ann_vol=("daily_returns", lambda r: r.std(ddof=0) * np.sqrt(252)),
        ann_sharpe=("daily_returns", sharpe_daily),
        max_dd=("daily_returns", max_drawdown_from_returns)
    )
    .round(3)
)

print("\n=== Volatility Regime Summary ===")
print(regime_stats)
# strategy: 100% invested in low and mid, 50% in high
analysis_df["position"] = np.where(analysis_df["regime"] == 2, 0.5, 1.0)
analysis_df["strategy_returns"] = analysis_df["daily_returns"] * analysis_df["position"]

# Buy & Hold benchmark
bh_sharpe = sharpe_daily(analysis_df["daily_returns"])
strat_sharpe = sharpe_daily(analysis_df["strategy_returns"])

bh_dd = max_drawdown_from_returns(analysis_df["daily_returns"])
strat_dd = max_drawdown_from_returns(analysis_df["strategy_returns"])

print("\n=== Backtest Summary ===")
print(f"Buy & Hold Sharpe : {bh_sharpe:.3f}")
print(f"Strategy Sharpe   : {strat_sharpe:.3f}")
print(f"Buy & Hold MaxDD  : {bh_dd:.1%}")
print(f"Strategy MaxDD    : {strat_dd:.1%}")


# ------------------ graph ------------------
def plot_regime_blocks(df, price_col="close", regime_col="regime"):
    regime_colors = {0: "#a8e6cf", 1: "#fff3b0", 2: "#ffaaa5"}
    regime_names = {0: "Low Vol", 1: "Mid Vol", 2: "High Vol"}

    # create block IDs whenever the regime changes
    block_id = df[regime_col].ne(df[regime_col].shift()).cumsum()

    blocks = []
    for _, seg in df.groupby(block_id):
        start, end = seg.index[0], seg.index[-1]
        r = seg[regime_col].iloc[0]
        if np.isnan(r):  # skip NaN zones
            continue
        blocks.append((start, end, int(r)))

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(df.index, df[price_col], color="black", linewidth=1.3, label="Close Price")

    # shade each block once
    for start, end, regime in blocks:
        ax.axvspan(start, end, color=regime_colors[regime], alpha=0.25,
                   label=regime_names[regime])

    # deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", fontsize=10)

    ax.set_title("SPY 100 Price and Volatility Tracker over the past 20 years", fontsize=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("#Price ($)", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_regime_blocks(analysis_df)





