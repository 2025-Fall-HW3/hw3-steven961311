"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        assets = self.price.columns[self.price.columns != self.exclude]

        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        ret_lb = 126      
        vol_lb = 63      
        ma_lb = 200     
        top_k = 4   
        base_target_ann_vol = 0.12 

        target_ann_vol = base_target_ann_vol / (1 + max(self.gamma, 0))

        start_idx = ma_lb

        for t in range(start_idx, len(self.price)):
            date = self.price.index[t]

            window_prices_mom = self.price.iloc[t - ret_lb : t]
            window_returns_vol = self.returns.iloc[t - vol_lb : t]

            momentum = (
                window_prices_mom[assets].iloc[-1] / window_prices_mom[assets].iloc[0]
                - 1
            )

            vol = window_returns_vol[assets].std()
            vol = vol.replace(0, np.nan)

            ma_window = self.price[assets].iloc[t - ma_lb : t].mean()
            current_price = self.price[assets].iloc[t]
            in_uptrend = current_price > ma_window

            score = (momentum.clip(lower=0) / vol)
            score = score.replace([np.inf, -np.inf], 0).fillna(0)

            score[~in_uptrend] = 0

            if score.sum() <= 0:
                self.portfolio_weights.loc[date, assets] = 0.0
                self.portfolio_weights.loc[date, self.exclude] = 0.0
                continue

            positive_assets = score[score > 0]
            top_k_effective = min(top_k, len(positive_assets))
            if top_k_effective == 0:
                self.portfolio_weights.loc[date, assets] = 0.0
                self.portfolio_weights.loc[date, self.exclude] = 0.0
                continue

            top_assets = positive_assets.nlargest(top_k_effective).index
            w_raw = positive_assets.loc[top_assets]

            w_raw = w_raw / w_raw.sum()

            avg_daily_vol = vol[top_assets].mean()
            if pd.isna(avg_daily_vol) or avg_daily_vol == 0:
                risk_scaler = 1.0
            else:
                ann_vol = avg_daily_vol * np.sqrt(252)
                if ann_vol <= 0 or pd.isna(ann_vol):
                    risk_scaler = 1.0
                else:
                    risk_scaler = min(1.0, target_ann_vol / ann_vol)

            w_scaled = w_raw * risk_scaler 

            self.portfolio_weights.loc[date, :] = 0.0
            self.portfolio_weights.loc[date, top_assets] = w_scaled.values
            self.portfolio_weights.loc[date, self.exclude] = 0.0

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
