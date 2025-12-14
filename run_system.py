#!/usr/bin/env python3
"""
Main execution script for Mega India Quant System.
Runs data collection, model training, forecasting, and backtesting.
"""

import sys
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

from mega_india_quant.main_engine import MegaIndiaQuantSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Mega India Quant System')
    parser.add_argument('--mode', default='backtest', choices=['backtest', 'live'],
                       help='Operating mode')
    parser.add_argument('--end-date', default=None,
                       help='End date for data collection (YYYY-MM-DD)')
    parser.add_argument('--output', default='./results/forecast.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("MEGA INDIA QUANT SYSTEM - INSTITUTIONAL TRADING ENGINE")
    logger.info("=" * 80)
    
    # Initialize system
    system = MegaIndiaQuantSystem(mode=args.mode)
    
    # Step 1: Fetch data
    logger.info("\n[STEP 1] FETCHING DATA")
    logger.info("-" * 80)
    
    if not system.fetch_all_data(args.end_date):
        logger.error("Data collection failed. Exiting.")
        return 1
    
    logger.info("✓ Data collection completed")
    
    # Step 2: Train models
    logger.info("\n[STEP 2] TRAINING MODELS")
    logger.info("-" * 80)
    
    if not system.train_all_models():
        logger.warning("Some models failed to train, continuing with available models")
    
    logger.info("✓ Model training completed")
    
    # Step 3: Generate forecast
    logger.info("\n[STEP 3] GENERATING FORECAST")
    logger.info("-" * 80)
    
    forecast = system.generate_unified_forecast()
    
    if forecast:
        logger.info(f"✓ Forecast generated")
        logger.info(f"  Final Signal: {forecast.get('final_signal', 0.5):.3f}")
        logger.info(f"  Confidence: {forecast.get('confidence', 0.5):.3f}")
        logger.info(f"  Regime: {forecast.get('regime', 'Unknown')}")
    else:
        logger.error("Forecast generation failed")
    
    # Step 4: Backtest
    logger.info("\n[STEP 4] RUNNING BACKTEST")
    logger.info("-" * 80)
    
    # Create synthetic backtest data
    if 'indian_equities' in system.macro_data_cache:
        equity_data = system.macro_data_cache['indian_equities']
        
        if len(equity_data) > 100 and 'NIFTY50' in equity_data.columns:
            # Use actual data for backtest
            backtest_data = equity_data[['NIFTY50']].copy()
            backtest_data.columns = ['NIFTY50']
            
            backtest_results = system.run_backtest(backtest_data)
            
            if backtest_results:
                summary = backtest_results.get('summary', {})
                logger.info("✓ Backtest completed")
                logger.info(f"  Initial Capital: ${summary.get('initial_capital', 0):.2f}")
                logger.info(f"  Final Capital: ${summary.get('final_capital', 0):.2f}")
                logger.info(f"  Total Return: {summary.get('total_return', 0):.2f}%")
                logger.info(f"  Sharpe Ratio: {summary.get('sharpe_ratio', 0):.3f}")
                logger.info(f"  Max Drawdown: {summary.get('max_drawdown', 0):.2f}%")
                logger.info(f"  Win Rate: {summary.get('win_rate', 0):.2f}%")
                logger.info(f"  Num Trades: {summary.get('num_trades', 0)}")
    
    # Step 5: Save results
    logger.info("\n[STEP 5] SAVING RESULTS")
    logger.info("-" * 80)
    
    system.save_results(args.output)
    logger.info(f"✓ Results saved to {args.output}")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("SYSTEM EXECUTION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    
    # Print final forecast summary
    logger.info("\nFINAL FORECAST SUMMARY:")
    logger.info("-" * 80)
    
    if forecast:
        logger.info(f"Macro Signal:           {forecast.get('macro_signal', 0.5):.3f}")
        logger.info(f"Factor Signal:          {forecast.get('factor_signal', 0.5):.3f}")
        logger.info(f"Technical Signal:       {forecast.get('technical_signal', 0.5):.3f}")
        logger.info(f"Deep Learning Signal:   {forecast.get('dl_signal', 0.5):.3f}")
        logger.info(f"Computer Vision Signal: {forecast.get('cv_signal', 0.5):.3f}")
        logger.info(f"Microstructure Signal:  {forecast.get('microstructure_signal', 0.5):.3f}")
        logger.info(f"\nFinal Composite Signal: {forecast.get('final_signal', 0.5):.3f}")
        logger.info(f"Forecast Confidence:    {forecast.get('confidence', 0.5)*100:.1f}%")
        logger.info(f"Market Regime:          {forecast.get('regime', 'Unknown')}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
