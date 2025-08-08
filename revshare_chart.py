import numpy as np
import matplotlib.pyplot as plt

# --- parameters -----------------------------------------------------------
tvl_cap_m = 250                       # range shown on the x‑axis (USD millions)
step_pct   = 0.425                    # %‑points of revenue per $1 M TVL
floor_pct  = 15                       # minimum share MegaETH always receives
full_pct   = 100                      # cap at 100 %
full_tvl_m = 200                      # TVL (USD millions) where share hits 100 %

# --- data -----------------------------------------------------------------
tvl_m = np.linspace(0, tvl_cap_m, tvl_cap_m + 1)        # x‑axis: 0 … 250 M
share_pct = np.minimum(full_pct, floor_pct + step_pct * tvl_m)

# --- plot -----------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(tvl_m, share_pct, linewidth=2)
plt.scatter([0, full_tvl_m], [floor_pct, full_pct])     # anchor points

# annotate the anchors
plt.annotate("15 %", (0, floor_pct),  textcoords="offset points", xytext=(10, -15))
plt.annotate("100 % at $200 M", (full_tvl_m, full_pct), textcoords="offset points",
             xytext=(10, 5))

# cosmetics
plt.xlabel("TVL committed (USD millions)")
plt.ylabel("Revenue share to MegaETH (%)")
plt.title("Revenue Share VS TVL Committed")
plt.grid(True)
plt.tight_layout()

# save a hi‑res copy and display
plt.savefig("revenue_share_schedule.png", dpi=300)
plt.show()
