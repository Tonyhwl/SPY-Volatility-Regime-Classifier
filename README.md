# SPY-Volatility-Regime-Classifier

A Python project that models S&P 500 volatility regimes and explores how market risk states impact trading strategies.  
Built to better understand how volatility clustering affects drawdowns, Sharpe ratios, and portfolio allocation.

---

## Overview

This project classifies the S&P 500 into low-, mid-, and high-volatility regimes using 30-day realised volatility.  
It then backtests a simple regime-based allocation strategy to see if adjusting exposure based on volatility can improve risk-adjusted returns.

I built this project to deepen my understanding of:

- Decision-making under uncertainty
- Volatility clustering and risk regimes
- Simple backtesting of allocation strategies

---

## Methodology

1. Data — 20+ years of daily S&P 500 prices.
2. Volatility Calculation — Rolling **30-day realised volatility** (annualised).
3. Regime Classification —  
   - Low vol  
   - Mid vol  
   - High vol  
   *(based on quantiles of rolling vol)*  
4. Backtesting — Compared a regime-aware allocation strategy vs buy-and-hold.

---

## Results

- High-volatility regimes (~10% of days) captured most major drawdowns (-43%), confirming strong volatility clustering.
- Backtested volatility-based allocation strategy:
    - Sharpe ratio: +26.2% improvement vs buy-and-hold
    - Max drawdown: reduced by 47%
- Generated visualisations showing regime transitions and performance differences.

---
- **Python** — data analysis
