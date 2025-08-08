import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load & daily-aggregate data
# ----------------------------
vol = (
    pd.read_csv("hype_vol.csv", parse_dates=["Date"])
      .rename(columns={"Date": "date", "Volume": "volume_usd"})
      .drop(columns=["Timestamp"], errors="ignore")
)

fee = (
    pd.read_csv("hl_fees.csv", parse_dates=["Date"])
      .rename(columns={"Date": "date", "Fees": "fees_usd"})
      .drop(columns=["Timestamp"], errors="ignore")
)

# Coerce to numeric in case of stray strings
vol["volume_usd"] = pd.to_numeric(vol["volume_usd"], errors="coerce")
fee["fees_usd"]   = pd.to_numeric(fee["fees_usd"], errors="coerce")

# Aggregate to daily (sum), then reindex to daily frequency
vol = (
    vol.groupby(vol["date"].dt.normalize(), as_index=True, sort=True)["volume_usd"]
       .sum()
       .to_frame()
       .asfreq("D")
)
fee = (
    fee.groupby(fee["date"].dt.normalize(), as_index=True, sort=True)["fees_usd"]
       .sum()
       .to_frame()
       .asfreq("D")
)

# ----------------------------
# Compute fee rate on overlap
# ----------------------------
overlap = vol.join(fee, how="inner")
overlap = overlap.replace([np.inf, -np.inf], np.nan).dropna()
overlap = overlap[overlap["volume_usd"] > 0]

if overlap.empty or len(overlap) < 14:
    raise ValueError("Not enough overlapping days with positive volume to estimate a fee rate.")

fee_rate = overlap["fees_usd"].sum() / overlap["volume_usd"].sum()

# ----------------------------
# Build daily revenue series
# ----------------------------
revenue_estimated = vol["volume_usd"] * fee_rate        # estimated revenue for ALL days
revenue_observed  = fee["fees_usd"]                     # observed (NaN when missing)
revenue_blended   = revenue_observed.combine_first(revenue_estimated)

# Scenarios (kept for completeness; not all are plotted)
revenue_100 = revenue_estimated           # exact HL volume path × fee_rate
revenue_50  = revenue_estimated * 0.5     # 50% of HL volume

# ----------------------------
# Assemble and save DAILY CSV (unchanged)
# ----------------------------
daily_df = pd.DataFrame({
    "volume_usd":            vol["volume_usd"],
    "revenue_observed_usd":  revenue_observed,
    "revenue_estimated_usd": revenue_estimated,
    "revenue_blended_usd":   revenue_blended,
    "revenue_100_usd":       revenue_100,
    "revenue_50_usd":        revenue_50
})
daily_df.to_csv("revenue_time_hl_match.csv")

# ----------------------------
# Aggregate to MONTHLY totals
# ----------------------------
monthly = daily_df.resample("ME").sum()

# For visual validation: observed fees per month (sum of actuals only)
monthly_observed = fee["fees_usd"].resample("ME").sum()
monthly_observed = monthly_observed.reindex(monthly.index)

# ----------------------------
# Save MONTHLY CSV
# ----------------------------
monthly_out = monthly.copy()
monthly_out["revenue_observed_usd"] = monthly_observed
monthly_out.to_csv("revenue_time_hl_match_monthly.csv")

# ----------------------------
# Formatting helper
# ----------------------------
def fmt_money_short(v):
    """Format USD to '$Xm' / '$X.Xm' or '$X.Xb'."""
    if pd.isna(v):
        return ""
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e9:
        s = f"{v/1e9:.1f}".rstrip("0").rstrip(".")
        return f"{sign}${s}b"
    else:
        s = f"{v/1e6:.1f}".rstrip("0").rstrip(".")
        return f"{sign}${s}m"


def add_staggered_bar_labels_above(ax, bars, labels, high_pts=12, low_pts=6,
                                   fontsize=9, start_high=True):
    """
    Place one label per bar, alternating vertical offsets *above* the bar top.
    Offsets are in points so spacing looks consistent regardless of scale.
    Labels are centered horizontally over each bar.
    """
    import matplotlib.transforms as mtransforms
    fig = ax.figure
    for i, (rect, lab) in enumerate(zip(bars, labels)):
        if not lab:
            continue
        h = rect.get_height()
        y_top = rect.get_y() + (h if h >= 0 else 0)
        x = rect.get_x() + rect.get_width() / 2

        use_high = (i % 2 == 0) if start_high else (i % 2 == 1)
        dy_pts = (high_pts if use_high else low_pts) / 72.0

        trans = ax.transData + mtransforms.ScaledTranslation(0, dy_pts, fig.dpi_scale_trans)
        ax.text(x, y_top, lab, ha="center", va="bottom",
                transform=trans, fontsize=fontsize, clip_on=False)

# ----------------------------
# Cumulative revenue (no MegaETH share; just total revenue)
# ----------------------------
monthly_out = monthly.copy()
monthly_out["revenue_observed_usd"] = fee["fees_usd"].resample("ME").sum().reindex(monthly.index)
monthly_out["cumulative_revenue_usd"] = monthly_out["revenue_blended_usd"].cumsum()
monthly_out.to_csv("revenue_time_hl_match_monthly.csv")

# Add cumulative revenue to the monthly table (total revenue to date)
monthly_out["cumulative_revenue_usd"] = monthly_out["revenue_blended_usd"].cumsum()
monthly_out.to_csv("revenue_time_hl_match_monthly.csv")  # update the CSV on disk





# ============================================================
# Scenarios: 100%, 50%, 10% of Hyperliquid revenue
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

SCENARIOS = [("100", 1.0), ("50", 0.5), ("10", 0.1)]

# Precompute X axis once
x = np.arange(len(monthly_out.index))
labels_x = [d.strftime("%Y-%b") for d in monthly_out.index]
width = 0.92  # centered bars with consistent gaps

# Add scenario columns to monthly_out and write a consolidated CSV
for tag, factor in SCENARIOS:
    monthly_out[f"revenue_{tag}_usd"] = monthly_out["revenue_blended_usd"] * factor
    monthly_out[f"cumulative_revenue_{tag}_usd"] = monthly_out[f"revenue_{tag}_usd"].cumsum()

monthly_out.to_csv("revenue_time_hl_match_monthly_scenarios.csv")

# Helper for 6-month markers
every = 6
marker_idx = np.arange(every-1, len(monthly_out), every)  # 5, 11, 17, ...
months_count = np.arange(1, len(monthly_out)+1)          # 1, 2, 3, ...

for tag, factor in SCENARIOS:
    # ----------------------------
    # Chart A: Monthly revenue (bars with labels)
    # ----------------------------
    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    heights_m = monthly_out[f"revenue_{tag}_usd"] / 1e6
    bars = ax.bar(x, heights_m, width=width, align="center",
                  label=f"Monthly revenue ({tag}% of HL)")

    # Labels above each bar, staggered to avoid overlap
    bar_labels = [fmt_money_short(v) for v in monthly_out[f"revenue_{tag}_usd"].values]
    add_staggered_bar_labels_above(ax, bars, bar_labels,
                                   high_pts=12, low_pts=6, fontsize=9, start_high=True)

    # Only show observed HL fee dots on the 100% chart (they reflect HL's data)
    if factor == 1.0 and "revenue_observed_usd" in monthly_out.columns:
        monthly_observed = monthly_out["revenue_observed_usd"]
        mask_obs = monthly_observed.notna() & (monthly_observed > 0)
        ax.plot(x[mask_obs], (monthly_observed[mask_obs] / 1e6).values,
                marker="o", linestyle="None", alpha=0.8)

    # Axis cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, rotation=45, ha="right")
    ax.set_xlim(-width/2, (len(x)-1) + width/2)
    ax.margins(y=0.20)
    ax.set_ylabel("Revenue (USD millions per month)")
    ax.set_xlabel("Month")
    ax.set_title(f"Monthly revenue – {tag}% of Hyperliquid volume path")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"revenue_monthly_bars_{tag}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ----------------------------
    # Chart B: Cumulative revenue (line) with 6‑month callouts
    # ----------------------------
    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    cum_m = monthly_out[f"cumulative_revenue_{tag}_usd"] / 1e6
    ax.plot(monthly_out.index, cum_m, linewidth=2, label=f"Cumulative revenue ({tag}% of HL)")

    # Markers and high-placed labels every 6 months
    ax.scatter(monthly_out.index[marker_idx], cum_m.iloc[marker_idx], s=40, zorder=5, color='red')

    # Simple text labels above the markers
    for idx in marker_idx:
        label = f"{months_count[idx]} mo: {fmt_money_short(monthly_out[f'cumulative_revenue_{tag}_usd'].iloc[idx])}"
        ax.text(monthly_out.index[idx], cum_m.iloc[idx] * 1.05, label,
                ha="center", va="bottom", fontsize=9, rotation=45)

    # Axis cosmetics
    ax.set_ylabel("Cumulative revenue (USD millions)")
    ax.set_xlabel("Month")
    ax.set_title(f"Cumulative revenue – {tag}% of Hyperliquid volume path")
    ax.grid(True, axis="both", alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(f"revenue_cumulative_line_{tag}.png", dpi=300, bbox_inches="tight")
    plt.show()

print("Saved charts:")
for tag, _ in SCENARIOS:
    print(f"  • revenue_monthly_bars_{tag}.png")
    print(f"  • revenue_cumulative_line_{tag}.png")
print("Saved data: revenue_time_hl_match_monthly_scenarios.csv")



# ----------------------------
# Console summary
# ----------------------------
print(f"Volume-weighted fee rate = {fee_rate*1e4:.2f} bps ({fee_rate:.5%})")
print(f"Overlap window: {overlap.index.min().date()} → {overlap.index.max().date()} "
      f"({len(overlap)} days)")
print("Outputs:")
print("  • revenue_time_hl_match.csv (daily)")
print("  • revenue_time_hl_match_monthly.csv (monthly)")
print("  • revenue_time_hl_match_monthly.png (bar chart)")

# Report summary statistics
print(f"Total revenue over period: {fmt_money_short(monthly_out['revenue_blended_usd'].sum())}")
print(f"Average monthly revenue: {fmt_money_short(monthly_out['revenue_blended_usd'].mean())}")
print(f"Peak monthly revenue: {fmt_money_short(monthly_out['revenue_blended_usd'].max())}")
print("Saved: revenue_monthly_bars.png and revenue_cumulative_line.png")

