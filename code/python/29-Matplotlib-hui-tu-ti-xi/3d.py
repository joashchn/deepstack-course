"""3D 演示：咖啡店月利润随定价与广告投入变化的曲面。

subplot 的 projection="3d" 把 Axes 升级成三维展格；
展板还是那张展板（Figure），只是多了一根 z 轴。
跑起来后可以用鼠标拖拽旋转视角。
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.sans-serif": ["PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC"],
        "axes.unicode_minus": False,
        "figure.dpi": 100,
    }
)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

# 两个自变量：咖啡定价（元/杯）与月广告投入（千元）
price = np.linspace(12, 30, 80)
ads = np.linspace(0, 8, 80)
P, A = np.meshgrid(price, ads)

# 模拟月利润（千元）：定价偏离 18 元、投放偏离 4 千元，利润都会下滑
demand = 8000 * np.exp(-((P - 18) ** 2) / 40) * np.exp(-((A - 4) ** 2) / 6)
profit = demand * (P - 8) / 1000 - A * 1.5

surface = ax.plot_surface(P, A, profit, cmap="viridis", alpha=0.95)

ax.set_xlabel("定价（元/杯）")
ax.set_ylabel("月广告（千元）")
ax.set_zlabel("月利润（千元）")  # type: ignore[attr-defined]
ax.set_title("咖啡店利润曲面：定价 × 广告的最优组合")
ax.view_init(elev=28, azim=-58)  # 挑一个顺眼的观察角度

fig.colorbar(surface, shrink=0.6, pad=0.1, label="月利润（千元）")

plt.show()
