"""交互演示四：CheckButtons 控制多条曲线的显隐。

三家门店各一条日销量曲线，勾选/取消勾选即时切换可见性。
回调收到被点击那一项的名字，拿它找到对应 Artist，
set_visible 取反——多系列对比图的"遥控器"。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons

plt.rcParams.update(
    {
        "font.sans-serif": ["PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC"],
        "axes.unicode_minus": False,
        "figure.dpi": 100,
    }
)

days = np.arange(1, 15)
shop_data = {
    "西湖店": 120 + 6 * days + np.random.default_rng(1).normal(0, 8, 14),
    "滨江店": 95 + 4 * days + np.random.default_rng(2).normal(0, 8, 14),
    "城西店": 150 + 2 * days + np.random.default_rng(3).normal(0, 8, 14),
}

fig, ax = plt.subplots(figsize=(8, 5))
fig.subplots_adjust(left=0.28)   # 左侧腾出放复选框的地方

lines: dict[str, Line2D] = {}
for name, values in shop_data.items():
    (lines[name],) = ax.plot(days, values, lw=2, label=name)
ax.legend(loc="upper left")
ax.set_title("门店日销量：勾选控制显隐")
ax.set_xlabel("天")
ax.set_ylabel("销量（杯）")

check_ax = fig.add_axes((0.05, 0.4, 0.18, 0.2))
check = CheckButtons(check_ax, list(shop_data), [True, True, True])


def toggle(name: str) -> None:
    """被点击的门店：可见性取反。"""
    lines[name].set_visible(not lines[name].get_visible())
    fig.canvas.draw_idle()


check.on_clicked(toggle)

plt.show()
