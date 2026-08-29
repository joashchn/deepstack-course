"""本章演示：一块 2x2 经营看板，把 Figure / Axes / Artist 三层结构"指给你看"。

整张看板是一块展板（Figure），被划成四个展格（Axes），
每个展格里摆着不同的展品（Artist：折线、柱子、散点、直方图）。
画面上的黄色标签是"导览牌"，逐一指认三层结构各自的位置。
"""

import numpy as np
import matplotlib.pyplot as plt

# 中文显示：macOS 优先 PingFang SC，其他系统按顺序回退
plt.rcParams.update(
    {
        "font.sans-serif": ["PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC"],
        "axes.unicode_minus": False,
        "figure.dpi": 100,
    }
)

rng = np.random.default_rng(42)

# ---------- 展板与四个展格 ----------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Figure：热浪咖啡年度经营看板", fontsize=16, fontweight="bold")

# ---------- 展格 1：折线（月销售额） ----------
months = np.arange(1, 13)
monthly_sales = np.array([21, 19, 24, 27, 31, 36, 39, 38, 33, 29, 26, 32])
axes[0, 0].plot(months, monthly_sales, marker="o", lw=2, color="royalblue")
axes[0, 0].set_title("Axes：月销售额（万元）")
axes[0, 0].set_xlabel("月份")

# ---------- 展格 2：柱状（季度杯量） ----------
quarters = ["一季度", "二季度", "三季度", "四季度"]
quarter_cups = [55, 94, 110, 87]  # 百杯
axes[0, 1].bar(quarters, quarter_cups, color="tomato", edgecolor="black", alpha=0.85)
axes[0, 1].set_title("Axes：季度杯量（百杯）")

# ---------- 展格 3：散点（气温 vs 热饮订单） ----------
daily_temp = np.linspace(-2, 36, 120)
hot_orders = 340 - 7.5 * daily_temp + rng.normal(0, 12, 120)
axes[1, 0].scatter(daily_temp, hot_orders, s=28, alpha=0.7, color="seagreen")
axes[1, 0].set_title("Axes：气温 vs 热饮订单")
axes[1, 0].set_xlabel("日均气温（℃）")
axes[1, 0].set_ylabel("热饮订单（单）")

# ---------- 展格 4：直方图（单笔消费金额） ----------
spend = rng.normal(38, 9, 3000)
axes[1, 1].hist(spend, bins=40, color="mediumpurple", edgecolor="white")
axes[1, 1].set_title("Axes：单笔消费金额分布（元）")

# ---------- 导览牌：指认 Artist 与 Axis ----------
# 1) 折线本身就是一个 Artist 对象
axes[0, 0].annotate(
    "Artist",
    xy=(7, 39),            # 箭头指向暑期峰值那个点
    xytext=(9, 20),        # 文字摆放的位置
    arrowprops={"arrowstyle": "->", "color": "black", "lw": 2},
    fontsize=12,
    bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.75},
)

# 2) 每根坐标轴也是 Artist，用"坐标轴分数"定位：0~1 的比例坐标
axes[0, 1].annotate(
    "axis",
    xy=(0, 0.4),           # 指向左侧纵轴
    xytext=(0.3, 0.15),
    xycoords="axes fraction",
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->", "color": "black", "lw": 2},
    fontsize=12,
    bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.75},
)

axes[0, 1].annotate(
    "axis",
    xy=(0.3, 0),           # 指向底部横轴
    xytext=(0.3, 0.15),
    xycoords="axes fraction",
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->", "color": "black", "lw": 2},
    ha="center",
    fontsize=12,
    bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.75},
)

plt.tight_layout()
plt.show()
