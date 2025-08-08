import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Import libraries (already done above)

# Step 2: Load the CSV files
# Note: The actual column names are different from the specification
fees = pd.read_csv("hl_fees.csv", parse_dates=["Date"])
vol = pd.read_csv("hype_vol.csv", parse_dates=["Date"])

# Rename columns to match the expected format
fees = fees.rename(columns={"Date": "date", "Fees": "fees_usd"})
vol = vol.rename(columns={"Date": "date", "Volume": "volume_usd"})

# Step 3: Inner-join on the date column
df = vol.merge(fees, on="date", how="inner")

# Step 4: Compute volume-weighted fee rate
fee_rate = df["fees_usd"].sum() / df["volume_usd"].sum()

# Step 5: Compute Hyperliquid's reference average daily volume
hl_daily_vol = df["volume_usd"].mean()

# Step 6: Define function for predicted daily volume
def volume_from_tvl(tvl, beta, anchor_vol=hl_daily_vol, anchor_tvl=400e6):
    alpha = anchor_vol / (anchor_tvl ** beta)
    return alpha * (tvl ** beta)

# Step 7: TVL grid
tvl_grid = np.linspace(0, 600e6, 601)  # 0 to 600M

# Step 8: Calculate projections for different beta values
results = []

for beta in [1.0, 0.8, 0.6]:
    v_pred = volume_from_tvl(tvl_grid, beta)
    rev_pred = v_pred * fee_rate * 365  # annualized
    share = np.minimum(1, 0.15 + 0.00425 * (tvl_grid / 1e6))
    megaeth_take = rev_pred * share
    
    # Store results
    for i, tvl in enumerate(tvl_grid):
        results.append({
            'tvl_usd': tvl,
            'beta': beta,
            'revenue_usd': rev_pred[i],
            'megaeth_take_usd': megaeth_take[i]
        })

# Create dataframe and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("tvl_revenue_projection.csv", index=False)

# Step 10: Create the plot
plt.figure(figsize=(12, 8))

# Create primary axis for total revenue
ax1 = plt.gca()
ax2 = ax1.twinx()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
betas = [1.0, 0.8, 0.6]

for i, beta in enumerate(betas):
    beta_data = results_df[results_df['beta'] == beta]
    
    # Plot total revenue (solid lines)
    ax1.plot(beta_data['tvl_usd'] / 1e6, beta_data['revenue_usd'] / 1e6, 
             color=colors[i], linewidth=2, label=f'β={beta} (Total Revenue)')
    
    # Plot MegaETH take (dashed lines)
    ax2.plot(beta_data['tvl_usd'] / 1e6, beta_data['megaeth_take_usd'] / 1e6,
             color=colors[i], linewidth=2, linestyle='--', alpha=0.7, 
             label=f'β={beta} (MegaETH Take)')

# Annotate anchor point (400M TVL)
anchor_tvl = 400e6
anchor_vol_beta_1 = volume_from_tvl(anchor_tvl, 1.0)
anchor_rev_beta_1 = anchor_vol_beta_1 * fee_rate * 365
ax1.scatter(anchor_tvl / 1e6, anchor_rev_beta_1 / 1e6, color='red', s=100, zorder=5)
ax1.annotate(f'Anchor: ${anchor_rev_beta_1/1e6:.1f}M\nat $400M TVL', 
             (anchor_tvl / 1e6, anchor_rev_beta_1 / 1e6),
             xytext=(10, 10), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Customize axes
ax1.set_xlabel('TVL (USD millions)', fontsize=12)
ax1.set_ylabel('Annual Revenue (USD millions)', fontsize=12, color='black')
ax2.set_ylabel('MegaETH Take (USD millions)', fontsize=12, color='gray')

# Set grid and layout
ax1.grid(True, alpha=0.3)
plt.title('TVL Revenue Projection Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Save the plot
plt.savefig("tvl_revenue_projection.png", dpi=300, bbox_inches='tight')
plt.show()

# Step 11: Print summary
print(f"\n=== TVL Revenue Projection Analysis ===")
print(f"Fee Rate: {fee_rate:.6f} ({fee_rate*100:.4f}%)")
print(f"Hyperliquid Daily Volume: ${hl_daily_vol:,.0f}")
print(f"Analysis Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
print(f"Total Days Analyzed: {len(df)}")

print(f"\n=== Beta (β) Explanation ===")
print("β represents the elasticity of volume with respect to TVL:")
print("• β = 1.0: Linear relationship - volume scales proportionally with TVL")
print("• β = 0.8: Sub-linear relationship - volume increases but with diminishing returns")
print("• β = 0.6: Strong diminishing returns - volume increases slowly with TVL")
print("\nLower β values represent more conservative volume projections.") 