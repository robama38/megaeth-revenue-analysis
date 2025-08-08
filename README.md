# MegaETH Revenue Analysis

This repository contains comprehensive revenue analysis for MegaETH based on Hyperliquid's volume trajectory and fee structure.

## Overview

The analysis projects revenue potential for MegaETH by replicating Hyperliquid's volume path and applying a volume-weighted fee rate calculated from historical data.

## Key Findings

- **Volume-weighted fee rate**: 2.67 bps (0.02670%)
- **Analysis period**: 2024-12-23 to 2025-07-19 (207 days)
- **Total revenue over period**: $592.9m (100% scenario)
- **Average monthly revenue**: $19.1m (100% scenario)
- **Peak monthly revenue**: $93.1m (100% scenario)

## Files

### Data Files
- `hl_fees.csv` - Hyperliquid daily fee data
- `hype_vol.csv` - Hyperliquid daily volume data
- `revenue_time_hl_match.csv` - Daily revenue projections
- `revenue_time_hl_match_monthly.csv` - Monthly aggregated data
- `revenue_time_hl_match_monthly_scenarios.csv` - Monthly data with all scenario projections

### Analysis Scripts
- `revenue_time_hl_match.py` - Main revenue analysis script
- `tvl_revenue_analysis.py` - TVL-based revenue projection analysis
- `revshare_chart.py` - Revenue share schedule visualization

### Charts Generated
- `revenue_monthly_bars_100.png` - Monthly revenue bars (100% HL scenario)
- `revenue_cumulative_line_100.png` - Cumulative revenue (100% HL scenario)
- `revenue_monthly_bars_50.png` - Monthly revenue bars (50% HL scenario)
- `revenue_cumulative_line_50.png` - Cumulative revenue (50% HL scenario)
- `revenue_monthly_bars_10.png` - Monthly revenue bars (10% HL scenario)
- `revenue_cumulative_line_10.png` - Cumulative revenue (10% HL scenario)
- `tvl_revenue_projection.png` - TVL vs revenue projection chart
- `revenue_share_schedule.png` - Revenue share schedule visualization

## Scenarios Analyzed

1. **100% HL Volume** - Replicating Hyperliquid's exact volume trajectory
2. **50% HL Volume** - Conservative scenario with half the volume
3. **10% HL Volume** - Very conservative scenario with 10% of volume

## Revenue Share Model

The analysis includes a revenue share schedule where MegaETH receives:
- Minimum 15% share at $0M TVL
- Increases by 0.425% per $1M TVL
- Caps at 100% at $200M TVL

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib
   ```

## Usage

Run the main analysis:
```bash
python revenue_time_hl_match.py
```

This will generate:
- Daily and monthly CSV files with revenue projections
- Bar charts showing monthly revenue for each scenario
- Line charts showing cumulative revenue over time
- Summary statistics in the console

## Methodology

1. **Fee Rate Calculation**: Volume-weighted fee rate calculated from overlapping period of HL fees and volume data
2. **Revenue Projection**: Applied fee rate to volume data to estimate revenue
3. **Scenario Analysis**: Created multiple scenarios based on different market penetration levels
4. **Revenue Share**: Incorporated MegaETH's revenue share model based on TVL

## Business Insights

- Significant revenue potential even with conservative market penetration
- Revenue scales linearly with volume in the base scenario
- Peak monthly revenues suggest strong seasonal patterns
- Cumulative analysis shows steady growth trajectory

## License

This project is for internal business analysis purposes.
