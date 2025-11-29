"""
Trading simulation and metrics evaluation for the full pipeline.

Calculates Hit Rate, Win/Loss Ratio, Total P&L, and Sharpe Ratio
on out-of-sample (test day) data.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_test_data(
    data_path: str = "data/gbm/gbm_predictions.parquet",
    test_day: str = "2025-10-08",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Load GBM predictions and filter to test day only for out-of-sample evaluation.
    
    Parameters
    ----------
    data_path : str
        Path to gbm_predictions.parquet
    test_day : str
        Test day to filter (YYYY-MM-DD format)
    logger : Logger, optional
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with test day data only
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} total rows from {data_path}")
    
    # Convert day column to string for comparison if needed
    if "day" in df.columns:
        df["day"] = pd.to_datetime(df["day"]).dt.strftime("%Y-%m-%d")
    elif "date" in df.columns:
        df["day"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    
    # Filter to test day
    df_test = df[df["day"] == test_day].copy()
    logger.info(f"Filtered to test day ({test_day}): {len(df_test):,} rows")
    
    return df_test


def compute_trade_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trade direction based on VWAP z-score.
    
    - Negative VWAP z-score: price below VWAP -> expect reversion UP -> LONG
    - Positive VWAP z-score: price above VWAP -> expect reversion DOWN -> SHORT
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feat_vwap_zscore column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with trade_direction column added (1=LONG, -1=SHORT)
    """
    df = df.copy()
    df["trade_direction"] = np.where(df["feat_vwap_zscore"] < 0, 1, -1)
    df["trade_direction_label"] = np.where(df["trade_direction"] == 1, "LONG", "SHORT")
    return df


def simulate_trades(
    df: pd.DataFrame,
    rf_threshold: float = 0.5,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Simulate trades using the full pipeline predictions.
    
    For each trade:
    - Entry: when GMM flags overextension and RF probability >= threshold
    - Direction: determined by VWAP z-score sign
    - Exit: based on triple-barrier outcome (label)
    - P&L: forward_return (already computed as label * ewm_std)
    
    Parameters
    ----------
    df : pd.DataFrame
        Test data with predictions
    rf_threshold : float
        RF probability threshold for taking trades (default 0.5)
    logger : Logger, optional
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with simulated trade outcomes
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    df = compute_trade_direction(df)
    
    # Filter trades by RF probability threshold
    df_trades = df[df["rf_proba_reversion"] >= rf_threshold].copy()
    logger.info(f"Trades with RF prob >= {rf_threshold}: {len(df_trades):,} / {len(df):,}")
    
    # Classify trade outcomes
    df_trades["outcome"] = df_trades["label"].map({
        1: "WIN",
        -1: "LOSS",
        0: "TIMEOUT"
    })
    
    # Compute directional P&L
    # For LONG: positive forward_return = profit
    # For SHORT: need to flip sign (but forward_return already accounts for direction)
    # Actually, forward_return = label * ewm_std is the magnitude, already signed
    df_trades["pnl"] = df_trades["forward_return"]
    
    # Add predicted exit levels for analysis
    df_trades["pred_stop"] = df_trades["gbm_pred_q10"]
    df_trades["pred_target"] = df_trades["gbm_pred_q50"]
    
    return df_trades


def compute_metrics(
    df_trades: pd.DataFrame,
    annualization_factor: float = 252.0,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Compute trading performance metrics.
    
    Metrics:
    - Hit Rate: wins / total trades
    - Win/Loss Ratio: avg win / abs(avg loss)
    - Total P&L: sum of all returns
    - Sharpe Ratio: mean return / std return * sqrt(N)
    
    Parameters
    ----------
    df_trades : pd.DataFrame
        Simulated trades DataFrame
    annualization_factor : float
        Factor for annualizing Sharpe (252 for daily)
    logger : Logger, optional
        Logger instance
        
    Returns
    -------
    dict
        Dictionary of computed metrics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    n_trades = len(df_trades)
    
    if n_trades == 0:
        logger.warning("No trades to evaluate")
        return {
            "n_trades": 0,
            "hit_rate": np.nan,
            "win_loss_ratio": np.nan,
            "total_pnl": 0.0,
            "sharpe_ratio": np.nan,
        }
    
    # Count outcomes
    wins = (df_trades["label"] == 1).sum()
    losses = (df_trades["label"] == -1).sum()
    timeouts = (df_trades["label"] == 0).sum()
    
    # Hit Rate
    hit_rate = wins / n_trades
    
    # Win/Loss Ratio
    win_returns = df_trades.loc[df_trades["label"] == 1, "pnl"]
    loss_returns = df_trades.loc[df_trades["label"] == -1, "pnl"]
    
    avg_win = win_returns.mean() if len(win_returns) > 0 else 0
    avg_loss = loss_returns.mean() if len(loss_returns) > 0 else 0
    
    if avg_loss != 0:
        win_loss_ratio = abs(avg_win / avg_loss)
    else:
        win_loss_ratio = np.inf if avg_win > 0 else np.nan
    
    # Total P&L
    total_pnl = df_trades["pnl"].sum()
    
    # Sharpe Ratio (per-trade basis, then annualized conceptually)
    # Using sqrt(N) scaling for intraday
    mean_return = df_trades["pnl"].mean()
    std_return = df_trades["pnl"].std()
    
    if std_return > 0:
        sharpe_ratio = (mean_return / std_return) * np.sqrt(n_trades)
    else:
        sharpe_ratio = np.nan
    
    # Additional statistics
    metrics = {
        "n_trades": n_trades,
        "n_wins": wins,
        "n_losses": losses,
        "n_timeouts": timeouts,
        "hit_rate": hit_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
        "total_pnl": total_pnl,
        "mean_pnl": mean_return,
        "std_pnl": std_return,
        "sharpe_ratio": sharpe_ratio,
        "max_win": win_returns.max() if len(win_returns) > 0 else np.nan,
        "max_loss": loss_returns.min() if len(loss_returns) > 0 else np.nan,
    }
    
    logger.info(f"Metrics computed: {n_trades} trades, {hit_rate:.1%} hit rate, "
                f"Sharpe={sharpe_ratio:.2f}, Total P&L={total_pnl:.6f}")
    
    return metrics


def compute_metrics_by_ticker(
    df_trades: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Compute performance metrics broken down by ticker.
    
    Parameters
    ----------
    df_trades : pd.DataFrame
        Simulated trades DataFrame
    logger : Logger, optional
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        Metrics per ticker
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    results = []
    
    for ticker in df_trades["ticker"].unique():
        df_ticker = df_trades[df_trades["ticker"] == ticker]
        n = len(df_ticker)
        
        if n == 0:
            continue
        
        wins = (df_ticker["label"] == 1).sum()
        losses = (df_ticker["label"] == -1).sum()
        
        hit_rate = wins / n if n > 0 else np.nan
        total_pnl = df_ticker["pnl"].sum()
        mean_pnl = df_ticker["pnl"].mean()
        std_pnl = df_ticker["pnl"].std()
        
        # Per-ticker Sharpe
        sharpe = (mean_pnl / std_pnl) * np.sqrt(n) if std_pnl > 0 else np.nan
        
        results.append({
            "ticker": ticker,
            "n_trades": n,
            "n_wins": wins,
            "n_losses": losses,
            "hit_rate": hit_rate,
            "total_pnl": total_pnl,
            "mean_pnl": mean_pnl,
            "sharpe": sharpe,
        })
    
    df_results = pd.DataFrame(results).sort_values("total_pnl", ascending=False)
    logger.info(f"Computed metrics for {len(df_results)} tickers")
    
    return df_results


def format_metrics_table(metrics: dict) -> str:
    """
    Format metrics dictionary as a markdown table for the report.
    
    Parameters
    ----------
    metrics : dict
        Metrics dictionary from compute_metrics()
        
    Returns
    -------
    str
        Markdown-formatted table
    """
    lines = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Trades | {metrics['n_trades']:,} |",
        f"| Wins | {metrics['n_wins']:,} |",
        f"| Losses | {metrics['n_losses']:,} |",
        f"| Timeouts | {metrics['n_timeouts']:,} |",
        f"| Hit Rate | {metrics['hit_rate']:.1%} |",
        f"| Avg Win | {metrics['avg_win']:.6f} |",
        f"| Avg Loss | {metrics['avg_loss']:.6f} |",
        f"| Win/Loss Ratio | {metrics['win_loss_ratio']:.2f} |",
        f"| Total P&L | {metrics['total_pnl']:.6f} |",
        f"| Mean P&L | {metrics['mean_pnl']:.6f} |",
        f"| Std P&L | {metrics['std_pnl']:.6f} |",
        f"| Sharpe Ratio | {metrics['sharpe_ratio']:.2f} |",
    ]
    return "\n".join(lines)


def format_ticker_table(df_ticker: pd.DataFrame) -> str:
    """
    Format ticker metrics as a markdown table for the report.
    
    Parameters
    ----------
    df_ticker : pd.DataFrame
        Ticker metrics from compute_metrics_by_ticker()
        
    Returns
    -------
    str
        Markdown-formatted table
    """
    lines = [
        "| Ticker | Trades | Wins | Losses | Hit Rate | Total P&L | Sharpe |",
        "|--------|--------|------|--------|----------|-----------|--------|",
    ]
    
    for _, row in df_ticker.iterrows():
        lines.append(
            f"| {row['ticker']} | {row['n_trades']} | {row['n_wins']} | "
            f"{row['n_losses']} | {row['hit_rate']:.1%} | "
            f"{row['total_pnl']:.6f} | {row['sharpe']:.2f} |"
        )
    
    return "\n".join(lines)
