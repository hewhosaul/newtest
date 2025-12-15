#!/usr/bin/env python3
"""
Test script to verify data collection works without errors.
"""

import logging
import sys
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_collection():
    """Test the data collection pipeline."""
    try:
        from mega_india_quant.data_feeds import MacroDataCollector
        
        logger.info("Testing data collection...")
        collector = MacroDataCollector()
        
        # Test date
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        logger.info(f"Testing with end_date: {end_date}")
        
        # Test US yields
        logger.info("\n[1] Testing US Yield Curve...")
        us_yields = collector.fetch_us_yield_curve(end_date)
        logger.info(f"  Result: {us_yields.shape}")
        logger.info(f"  Columns: {list(us_yields.columns)[:3]}...")
        assert len(us_yields) > 0, "US yields empty"
        assert us_yields.shape[1] > 0, "US yields no columns"
        logger.info("  ✓ Pass")
        
        # Test Indian yields
        logger.info("\n[2] Testing Indian Yield Curve...")
        ind_yields = collector.fetch_indian_yield_curve(end_date)
        logger.info(f"  Result: {ind_yields.shape}")
        assert len(ind_yields) > 0, "Indian yields empty"
        logger.info("  ✓ Pass")
        
        # Test Global PMI
        logger.info("\n[3] Testing Global PMI...")
        pmi = collector.fetch_global_pmi(end_date)
        logger.info(f"  Result: {pmi.shape}")
        assert len(pmi) > 0, "PMI empty"
        logger.info("  ✓ Pass")
        
        # Test Inflation
        logger.info("\n[4] Testing Inflation Data...")
        inflation = collector.fetch_inflation_data(end_date)
        logger.info(f"  Result: {inflation.shape}")
        assert len(inflation) > 0, "Inflation empty"
        logger.info("  ✓ Pass")
        
        # Test Credit Spreads
        logger.info("\n[5] Testing Credit Spreads...")
        spreads = collector.fetch_credit_spreads(end_date)
        logger.info(f"  Result: {spreads.shape}")
        assert len(spreads) > 0, "Spreads empty"
        logger.info("  ✓ Pass")
        
        # Test FX Data
        logger.info("\n[6] Testing FX Data...")
        fx = collector.fetch_fx_data(end_date)
        logger.info(f"  Result: {fx.shape}")
        assert len(fx) > 0, "FX data empty"
        logger.info("  ✓ Pass")
        
        # Test Commodities
        logger.info("\n[7] Testing Commodities...")
        comm = collector.fetch_commodity_data(end_date)
        logger.info(f"  Result: {comm.shape}")
        assert len(comm) > 0, "Commodities empty"
        logger.info("  ✓ Pass")
        
        # Test Semiconductors
        logger.info("\n[8] Testing Semiconductors...")
        semi = collector.fetch_semiconductor_data(end_date)
        logger.info(f"  Result: {semi.shape}")
        assert len(semi) > 0, "Semiconductors empty"
        logger.info("  ✓ Pass")
        
        # Test Indian Equities
        logger.info("\n[9] Testing Indian Equities...")
        eq = collector.fetch_indian_equity_data(end_date)
        logger.info(f"  Result: {eq.shape}")
        assert len(eq) > 0, "Equities empty"
        logger.info("  ✓ Pass")
        
        # Test Trade Indices
        logger.info("\n[10] Testing Trade Indices...")
        trade = collector.fetch_global_trade_indices(end_date)
        logger.info(f"  Result: {trade.shape}")
        assert len(trade) > 0, "Trade empty"
        logger.info("  ✓ Pass")
        
        # Test Liquidity
        logger.info("\n[11] Testing Liquidity Indicators...")
        liq = collector.fetch_liquidity_indicators(end_date)
        logger.info(f"  Result: {liq.shape}")
        assert len(liq) > 0, "Liquidity empty"
        logger.info("  ✓ Pass")
        
        # Test Indian Macro
        logger.info("\n[12] Testing Indian Macro Indicators...")
        macro = collector.fetch_indian_macro_indicators(end_date)
        logger.info(f"  Result: {macro.shape}")
        assert len(macro) > 0, "Macro empty"
        logger.info("  ✓ Pass")
        
        # Test collect all
        logger.info("\n[13] Testing Collect All...")
        all_data = collector.collect_all_macro_data(end_date)
        logger.info(f"  Collected {len(all_data)} data sources")
        
        valid_count = 0
        for key, value in all_data.items():
            if isinstance(value, type(us_yields)) and len(value) > 0:
                valid_count += 1
                logger.info(f"    ✓ {key}: {value.shape}")
        
        assert valid_count > 0, "No valid data sources"
        logger.info(f"  ✓ Pass ({valid_count} valid sources)")
        
        logger.info("\n" + "="*60)
        logger.info("✓ ALL DATA COLLECTION TESTS PASSED")
        logger.info("="*60)
        return 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == '__main__':
    sys.exit(test_data_collection())
