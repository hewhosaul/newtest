"""
Visualization Module: Creates beautiful dashboards and charts for backtest results.
Includes equity curve, drawdown, performance metrics, factor analysis, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available, using matplotlib only")


class BacktestVisualizer:
    """Create professional backtest visualizations."""
    
    def __init__(self, output_dir: str = './results/charts'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_equity_curve(self, equity_curve: List[float], 
                         save_path: str = None) -> None:
        """Plot equity curve over time."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        equity_array = np.array(equity_curve)
        dates = pd.date_range(end=datetime.now(), periods=len(equity_curve), freq='D')
        
        ax.plot(dates, equity_array, linewidth=2, color='#2E86AB', label='Equity Curve')
        ax.fill_between(dates, equity_array, alpha=0.3, color='#2E86AB')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'equity_curve.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved equity curve to {save_path}")
        plt.close()
    
    def plot_drawdown(self, equity_curve: List[float],
                     save_path: str = None) -> None:
        """Plot drawdown over time."""
        equity_array = np.array(equity_curve)
        cummax = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - cummax) / cummax * 100
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        dates = pd.date_range(end=datetime.now(), periods=len(equity_curve), freq='D')
        
        ax.fill_between(dates, drawdown, 0, color='#E63946', alpha=0.6, label='Drawdown')
        ax.plot(dates, drawdown, linewidth=1, color='#E63946')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'drawdown.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved drawdown to {save_path}")
        plt.close()
    
    def plot_performance_metrics(self, report: Dict, 
                                save_path: str = None) -> None:
        """Plot performance metrics as bars."""
        if 'risk_metrics' not in report:
            return
        
        metrics = report['risk_metrics']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Performance Metrics', fontsize=16, fontweight='bold')
        
        # Sharpe Ratio
        ax = axes[0, 0]
        sharpe = metrics.get('sharpe_ratio', 0)
        color = '#06D6A0' if sharpe > 1 else '#E63946'
        ax.bar(['Sharpe Ratio'], [sharpe], color=color, alpha=0.7)
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.3, label='Target (1.0)')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_ylim(bottom=0)
        ax.legend()
        
        # Max Drawdown
        ax = axes[0, 1]
        max_dd = abs(metrics.get('max_drawdown', 0)) * 100
        ax.bar(['Max Drawdown'], [max_dd], color='#E63946', alpha=0.7)
        ax.set_ylabel('Max Drawdown (%)')
        ax.set_ylim(bottom=0)
        
        # Win Rate
        ax = axes[1, 0]
        if 'trade_metrics' in report:
            win_rate = report['trade_metrics'].get('win_rate', 0) * 100
            ax.bar(['Win Rate'], [win_rate], color='#06D6A0', alpha=0.7)
            ax.set_ylabel('Win Rate (%)')
            ax.set_ylim([0, 100])
        
        # Volatility
        ax = axes[1, 1]
        vol = metrics.get('volatility', 0) * 100
        ax.bar(['Volatility'], [vol], color='#FFB703', alpha=0.7)
        ax.set_ylabel('Volatility (%)')
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'performance_metrics.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance metrics to {save_path}")
        plt.close()
    
    def plot_trade_analysis(self, report: Dict,
                           save_path: str = None) -> None:
        """Plot trade statistics."""
        if 'trade_metrics' not in report:
            return
        
        metrics = report['trade_metrics']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Trade Analysis', fontsize=16, fontweight='bold')
        
        # Win vs Lose
        ax = axes[0, 0]
        wins = metrics.get('winning_trades', 0)
        losses = metrics.get('losing_trades', 0)
        ax.bar(['Wins', 'Losses'], [wins, losses], color=['#06D6A0', '#E63946'], alpha=0.7)
        ax.set_ylabel('Count')
        
        # Profit Factor
        ax = axes[0, 1]
        pf = metrics.get('profit_factor', 0)
        color = '#06D6A0' if pf > 1 else '#E63946'
        ax.bar(['Profit Factor'], [pf], color=color, alpha=0.7)
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.3)
        ax.set_ylabel('Profit Factor')
        
        # Avg Winner vs Loser
        ax = axes[1, 0]
        avg_win = metrics.get('avg_winner', 0)
        avg_loss = metrics.get('avg_loser', 0)
        ax.bar(['Avg Winner %', 'Avg Loser %'], [avg_win, avg_loss], 
               color=['#06D6A0', '#E63946'], alpha=0.7)
        ax.set_ylabel('Return %')
        
        # Total Trades
        ax = axes[1, 1]
        total = metrics.get('total_trades', 0)
        ax.text(0.5, 0.5, f'{total}\nTotal Trades', 
               ha='center', va='center', fontsize=20, fontweight='bold',
               transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'trade_analysis.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved trade analysis to {save_path}")
        plt.close()
    
    def create_dashboard(self, report: Dict) -> None:
        """Create comprehensive dashboard."""
        self.plot_equity_curve(report['equity_curve'])
        self.plot_drawdown(report['equity_curve'])
        self.plot_performance_metrics(report)
        self.plot_trade_analysis(report)
        
        logger.info("Dashboard created successfully")
    
    def create_summary_report(self, report: Dict, 
                             save_path: str = None) -> None:
        """Create text summary report."""
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'backtest_report.txt')
        
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MEGA INDIA QUANT - BACKTEST REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary
            summary = report.get('summary', {})
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Initial Capital:    ${summary.get('initial_capital', 0):,.2f}\n")
            f.write(f"Final Capital:      ${summary.get('final_capital', 0):,.2f}\n")
            f.write(f"Total Return:       {summary.get('total_return', 0)*100:.2f}%\n")
            f.write(f"Annual Return:      {summary.get('annual_return', 0)*100:.2f}%\n")
            f.write(f"Total P&L:          ${summary.get('total_pnl', 0):,.2f}\n")
            f.write(f"Total Fees:         ${summary.get('total_fees', 0):,.2f}\n\n")
            
            # Risk Metrics
            risk = report.get('risk_metrics', {})
            f.write("RISK METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Sharpe Ratio:       {risk.get('sharpe_ratio', 0):.3f}\n")
            f.write(f"Sortino Ratio:      {risk.get('sortino_ratio', 0):.3f}\n")
            f.write(f"Max Drawdown:       {risk.get('max_drawdown', 0)*100:.2f}%\n")
            f.write(f"Calmar Ratio:       {risk.get('calmar_ratio', 0):.3f}\n")
            f.write(f"Volatility:         {risk.get('volatility', 0)*100:.2f}%\n\n")
            
            # Trade Metrics
            trades = report.get('trade_metrics', {})
            f.write("TRADE METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Trades:       {trades.get('total_trades', 0)}\n")
            f.write(f"Winning Trades:     {trades.get('winning_trades', 0)}\n")
            f.write(f"Losing Trades:      {trades.get('losing_trades', 0)}\n")
            f.write(f"Win Rate:           {trades.get('win_rate', 0)*100:.2f}%\n")
            f.write(f"Profit Factor:      {trades.get('profit_factor', 0):.3f}\n")
            f.write(f"Avg Trade Return:   {trades.get('avg_trade_return', 0):.2f}%\n")
            f.write(f"Avg Winner:         {trades.get('avg_winner', 0):.2f}%\n")
            f.write(f"Avg Loser:          {trades.get('avg_loser', 0):.2f}%\n")
        
        logger.info(f"Saved report to {save_path}")
