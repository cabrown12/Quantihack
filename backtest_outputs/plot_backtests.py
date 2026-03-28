import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE_DIR
PLOT_DIR = os.path.join(BASE_DIR, "backtest_plots")

os.makedirs(PLOT_DIR, exist_ok=True)

# ===============================
# LOAD RESULTS
# ===============================
rank_path = os.path.join(OUT_DIR, "all_backtests_ranked.csv")

if not os.path.exists(rank_path):
    raise FileNotFoundError("Run backtest first bro")

df = pd.read_csv(rank_path)

# ===============================
# 1. TOP SHARPE STRATEGIES
# ===============================
top = df.head(10).copy()

top["label"] = (
    top["signal"] + " | " +
    top["market"] + " | lag=" +
    top["lag"].astype(str)
)

plt.figure(figsize=(12, 6))
plt.barh(top["label"][::-1], top["sharpe"][::-1])
plt.title("Top Strategies by Sharpe")
plt.xlabel("Sharpe Ratio")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "top_sharpe.png"), dpi=200)
plt.close()

# ===============================
# 2. EQUITY CURVES (MOST IMPORTANT)
# ===============================
equity_files = glob.glob(os.path.join(OUT_DIR, "equity_*.csv"))

for f in equity_files:
    try:
        eq = pd.read_csv(f, parse_dates=["date"])
    except:
        continue

    if not {"cum_strategy", "cum_buyhold"}.issubset(eq.columns):
        continue

    name = os.path.basename(f).replace(".csv", "")

    plt.figure(figsize=(12, 5))
    plt.plot(eq["date"], eq["cum_strategy"], linewidth=2, label="Strategy")
    plt.plot(eq["date"], eq["cum_buyhold"], linewidth=2, label="Buy & Hold")
    plt.axhline(1.0, linestyle="--")
    plt.title(name.replace("_", " "))
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}.png"), dpi=200)
    plt.close()

# ===============================
# 3. HEATMAPS (lag vs threshold)
# ===============================
grid_files = glob.glob(os.path.join(OUT_DIR, "grid_*.csv"))

for f in grid_files:
    try:
        g = pd.read_csv(f)
    except:
        continue

    if not {"lag", "threshold", "sharpe"}.issubset(g.columns):
        continue

    pivot = g.pivot_table(
        index="lag",
        columns="threshold",
        values="sharpe",
        aggfunc="mean"
    )

    if pivot.empty:
        continue

    name = os.path.basename(f).replace(".csv", "")

    plt.figure(figsize=(8, 6))
    plt.imshow(pivot, aspect="auto")
    plt.colorbar(label="Sharpe")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title(name.replace("_", " "))
    plt.xlabel("Threshold")
    plt.ylabel("Lag")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}.png"), dpi=200)
    plt.close()

print("All plots saved in:", PLOT_DIR)