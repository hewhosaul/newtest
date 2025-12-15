"""
Supply Chain & Structural Models: TSMC → semiconductor prices → tech stocks → India exposure.
Models: Supply chain elasticity, input-output analysis, margin propagation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemiconductorSupplyChainModel:
    """
    Model semiconductor supply chain: TSMC wafer capacity → TSMC stock → TSMC ADR (in India exposure)
    → semiconductor ETFs → Indian tech stocks.
    """
    
    def __init__(self):
        self.tsmc_utilization = 0.85
        self.wafer_price_trend = 0
        self.semiconductor_cycle_stage = 'neutral'  # down, up, peak, trough
        self.days_in_cycle = 0
    
    def analyze_tsmc_capacity(self, 
                             wafer_starts: float,
                             capacity: float = 3000000) -> Dict[str, float]:
        """Analyze TSMC wafer capacity and utilization."""
        
        utilization = wafer_starts / capacity
        self.tsmc_utilization = utilization
        
        # Pricing power: high utilization → pricing power up
        pricing_power = np.clip(utilization - 0.70, -0.2, 0.3)
        
        # Margin impact: high util → higher margins
        margin_delta = pricing_power * 300  # basis points
        
        return {
            'utilization': utilization,
            'capacity_tight': utilization > 0.85,
            'pricing_power': pricing_power,
            'margin_delta_bps': margin_delta,
        }
    
    def model_wafer_pricing(self, 
                           wafer_index: float,
                           demand_growth: float) -> float:
        """Model wafer spot prices based on supply-demand."""
        
        # Wafer price ≈ capacity utilization * demand_growth
        base_price = 10000  # $/wafer baseline
        
        # Capacity tight → prices up
        price_multiplier = 1.0 + self.tsmc_utilization * demand_growth
        
        spot_price = base_price * price_multiplier
        
        self.wafer_price_trend = (spot_price - base_price) / base_price
        
        return spot_price
    
    def propagate_to_tech_stocks(self, 
                                wafer_price: float,
                                base_wafer_price: float = 10000) -> Dict[str, float]:
        """Propagate wafer price changes to tech stock margins."""
        
        # Wafer price change → cost increase
        wafer_price_delta = (wafer_price - base_wafer_price) / base_wafer_price
        
        # COGS sensitivity to wafer prices (typically 15-25% of COGS)
        cogs_sensitivity = 0.20
        cogs_delta = wafer_price_delta * cogs_sensitivity  # % change in COGS
        
        # If COGS up 5%, operating margin down by ~2-3%
        margin_compression = -cogs_delta * 0.5  # gross margin impact
        
        # Revenue: wafer price up typically signals strong demand → revenue +
        revenue_uplift = min(wafer_price_delta * 0.3, 0.1)  # capped at 10%
        
        # Net effect: EPS = (revenue + cost_up) / shares
        eps_delta = revenue_uplift - cogs_delta  # multiplicative
        
        # Map to stock price: typically 12-15x P/E, so 1% EPS → 0.8-1.0% price
        pe_multiple = 14
        price_sensitivity = eps_delta * (pe_multiple * 0.07)  # 7% per 1% EPS
        
        return {
            'cogs_delta': cogs_delta,
            'margin_compression_bps': margin_compression * 10000,
            'revenue_uplift': revenue_uplift,
            'eps_delta': eps_delta,
            'stock_price_delta': price_sensitivity,
        }
    
    def map_to_indian_tech_exposure(self,
                                   tsmc_price_delta: float,
                                   tcs_exposure: float = 0.15,
                                   infosys_exposure: float = 0.10,
                                   wipro_exposure: float = 0.12) -> Dict[str, float]:
        """
        Map TSMC/semiconductor cycle to Indian IT companies.
        TCS, Infosys, Wipro have 15-20% revenue from semiconductor design/fabless.
        """
        
        # IT services gain from semicond upswing (higher capex, design work)
        # But face pressure from margin compression (cost inflation)
        
        # Net exposure: ~60% of semiconductor cycle effect
        tcs_delta = tsmc_price_delta * tcs_exposure
        infosys_delta = tsmc_price_delta * infosys_exposure
        wipro_delta = tsmc_price_delta * wipro_exposure
        
        return {
            'TCS_delta': tcs_delta,
            'INFY_delta': infosys_delta,
            'WIPRO_delta': wipro_delta,
            'semiconductor_cycle_exposure': tcs_delta + infosys_delta + wipro_delta,
        }
    
    def update_cycle_stage(self):
        """Update semiconductor cycle stage based on trend."""
        self.days_in_cycle += 1
        
        # Cycle length: ~8 years = ~2000 trading days
        cycle_position = (self.days_in_cycle % 2000) / 2000
        
        if cycle_position < 0.25:
            self.semiconductor_cycle_stage = 'trough'
        elif cycle_position < 0.5:
            self.semiconductor_cycle_stage = 'up'
        elif cycle_position < 0.75:
            self.semiconductor_cycle_stage = 'peak'
        else:
            self.semiconductor_cycle_stage = 'down'


class SupplyChainInputOutputModel:
    """
    Input-output model: energy prices → power cos margins → industrial cos → all sectors.
    Similar to Leontief model.
    """
    
    def __init__(self):
        self.io_matrix = self._initialize_io_matrix()
    
    def _initialize_io_matrix(self) -> np.ndarray:
        """Initialize simple 3-sector IO matrix."""
        # Sectors: Energy, Materials, Manufacturing, IT Services
        # io[i,j] = how much sector j needs from sector i per unit output
        
        io_matrix = np.array([
            [0.1, 0.3, 0.25, 0.05],  # Energy consumption
            [0.05, 0.2, 0.3, 0.05],  # Materials consumption
            [0.1, 0.15, 0.2, 0.1],   # Manufacturing
            [0.05, 0.05, 0.05, 0.1],  # IT services
        ])
        
        return io_matrix
    
    def simulate_energy_shock(self, energy_price_delta: float) -> Dict[str, float]:
        """Simulate impact of energy price shock through supply chain."""
        
        # Initial shock to energy costs
        shocks = np.array([energy_price_delta, 0, 0, 0])
        
        # Leontief model: (I - A)^-1 * shocks
        identity = np.eye(4)
        try:
            leontief_inverse = np.linalg.inv(identity - self.io_matrix)
            total_impacts = leontief_inverse @ shocks
        except np.linalg.LinAlgError:
            total_impacts = shocks
        
        sectors = ['Energy', 'Materials', 'Manufacturing', 'IT_Services']
        return {sector: total_impacts[i] for i, sector in enumerate(sectors)}


class CommodityInflationModel:
    """Model commodity price → inflation → RBI policy → equity impact."""
    
    def __init__(self):
        self.oil_price = 80  # $/barrel
        self.copper_price = 9000  # $/tonne
        self.inflation_rate = 0.05
        self.rbi_rate = 0.06
    
    def model_inflation_from_commodities(self,
                                        oil_price: float,
                                        copper_price: float) -> float:
        """Model CPI from commodity prices."""
        
        # Oil contribution to inflation: ~30% of CPI in India
        oil_weight = 0.3
        oil_inflation = (oil_price - 80) / 80 * 0.5  # 50% pass-through
        
        # Commodities/materials: ~20% of CPI
        commodity_weight = 0.2
        copper_inflation = (copper_price - 9000) / 9000 * 0.4  # 40% pass-through
        
        # Core inflation: ~50%
        core_inflation = 0.03  # Anchored
        
        total_inflation = (oil_weight * oil_inflation + 
                          commodity_weight * copper_inflation + 
                          0.5 * core_inflation)
        
        self.inflation_rate = total_inflation
        return total_inflation
    
    def model_rbi_response(self, inflation: float, target: float = 0.04) -> float:
        """Taylor rule: RBI rate response to inflation."""
        
        # Simple Taylor: r = r_neutral + 1.5*(inflation - target) + 0.5*output_gap
        r_neutral = 0.05
        inflation_gap = inflation - target
        output_gap = -0.01  # Assuming slight slowdown
        
        new_rate = r_neutral + 1.5 * inflation_gap + 0.5 * output_gap
        self.rbi_rate = np.clip(new_rate, 0.02, 0.10)
        
        return self.rbi_rate
    
    def model_equity_impact(self,
                           rbi_rate_delta: float,
                           inflation_delta: float) -> Dict[str, float]:
        """Model equity valuation impact from inflation/rates."""
        
        # Discount rate effect: 1% rate up → ~5-7% equity valuation down
        pe_compression = -rbi_rate_delta * 60  # basis points
        
        # Inflation expectations impact:
        # - Real rates up → equities down (discount rate)
        # - Nominal growth up → equities benefit
        # - Net typically negative for high inflation
        inflation_discount = -inflation_delta * 40  # bps
        
        # Sector impacts:
        # - Rate-sensitive (Realty, Banks): negative
        # - Commodity-linked (Metals, Energy): positive (inflation hedge)
        # - Tech, Pharma: neutral to slightly negative
        
        return {
            'market_valuation_impact_bps': pe_compression + inflation_discount,
            'rate_sensitive_impact': pe_compression * 1.5,  # banks, realty
            'commodity_hedge_impact': inflation_delta * 100,
            'nominal_growth_impact': inflation_delta * 20,
        }
