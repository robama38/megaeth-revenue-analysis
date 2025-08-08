import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
#  Parameters you might tweak
# ----------------------------------------------------------------------
FEE_CSV       = "hl_fees.csv"
VOL_CSV       = "hype_vol.csv"
ANCHOR_TVL    = 400e6                # USD
BETA_LIST     = [1.00, 0.80, 0.60]   # elasticity scenarios
YEAR_DAYS     = 365                  # use 365.25 if you prefer
PLOT_TVL_MAX  = 600e6                # right‑hand end of x‑axis
SAVE_CSV      = "tvl_revenue_projection.csv"
SAVE_PNG      = "tvl_revenue_projection.png"

# ----------------------------------------------------------------------
#  1.  Load & pre‑tidy data
# ----------------------------------------------------------------------
fees = (pd.read_csv(FEE_CSV, parse_dates=["Date"])
          .rename(columns={"Date": "date", "Fees": "fees_usd"})
          .drop_duplicates(subset="date"))

vol  = (pd.read_csv(VOL_CSV,  parse_dates=["Date"])
          .rename(columns={"Date": "date", "Volume": "volume_usd"})
          .drop_duplicates(subset="date"))

df = (vol.merge(fees, on="date", how="inner")         # keeps only overlapping days
          .query("volume_usd > 0")                    # safety: avoid /0
          .sort_values("date"))

fee_rate = df.fees_usd.sum() / df.volume_usd.sum()    # volume‑weighted
hl_daily_vol = df.volume_usd.mean()

# ----------------------------------------------------------------------
#  2.  Helper functions
# ----------------------------------------------------------------------
def volume_from_tvl(tvl_usd, beta,
                    anchor_vol=hl_daily_vol, anchor_tvl=ANCHOR_TVL):
    """Project *daily* volume at a given TVL using a power law."""
    alpha = anchor_vol / (anchor_tvl ** beta)
    return alpha * (tvl_usd ** beta)

share_from_tvl = lambda tvl: np.minimum(1, 0.15 + 0.85 * tvl / 200e6)

# ----------------------------------------------------------------------
#  3.  Grid & projections
# ----------------------------------------------------------------------
tvl_grid = np.arange(0, PLOT_TVL_MAX + 1e6, 1e6)                 # 1 M steps
all_rows = []

for beta in BETA_LIST:
    vol_pred   = volume_from_tvl(tvl_grid, beta)
    rev_pred   = vol_pred * fee_rate * YEAR_DAYS                 # annual
    share      = share_from_tvl(tvl_grid)
    mega_take  = rev_pred * share

    all_rows.append(pd.DataFrame({
        "tvl_usd": tvl_grid,
        "beta": beta,
        "revenue_usd": rev_pred,
        "megaeth_take_usd": mega_take
    }))

results_df = pd.concat(all_rows, ignore_index=True)
results_df.to_csv(SAVE_CSV, index=False)

# ----------------------------------------------------------------------
#  4.  Plot
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()

for beta in BETA_LIST:
    subset = results_df.query("beta == @beta")
    ax1.plot(subset.tvl_usd / 1e6, subset.revenue_usd / 1e6,
             linewidth=2, label=f"β={beta} – Total rev.")
    ax2.plot(subset.tvl_usd / 1e6, subset.megaeth_take_usd / 1e6,
             linestyle="--", linewidth=2, label=f"β={beta} – MegaETH take")

# anchor annotation (use default colours)
anchor_rev = volume_from_tvl(ANCHOR_TVL, 1.0) * fee_rate * YEAR_DAYS
ax1.scatter(ANCHOR_TVL / 1e6, anchor_rev / 1e6, s=80, zorder=5)
ax1.annotate(f"${anchor_rev/1e6:,.1f} M\nat ${ANCHOR_TVL/1e6:,.0f} M TVL",
             (ANCHOR_TVL / 1e6, anchor_rev / 1e6),
             xytext=(10, 10), textcoords="offset points")

ax1.set_xlabel("TVL (USD millions)")
ax1.set_ylabel("Annual protocol revenue (USD millions)")
ax2.set_ylabel("Annual MegaETH take (USD millions)")
ax1.set_title("Projected annual revenue vs TVL\n(power‑law volume, HL fee‑rate)")
ax1.grid(True, alpha=0.3)

# unified legend
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left")

plt.tight_layout()
plt.savefig(SAVE_PNG, dpi=300)
plt.show()

# ----------------------------------------------------------------------
#  5.  Console summary
# ----------------------------------------------------------------------
print(f"Fee rate (volume‑weighted): {fee_rate*1e4:.2f} bps "
      f"({fee_rate:.5%})")
print(f"Average daily volume in sample: ${hl_daily_vol:,.0f}")
print(f"Sample: {df.date.min().date()} … {df.date.max().date()} "
      f"({len(df)} days)")
print("\nβ captures the elasticity of volume wrt TVL:")
print(" 1.0 = linear; 0.8 = diminishing; 0.6 = conservative.")
print(f"\nOutputs written to {SAVE_CSV} and {SAVE_PNG}")
