"""交互演示二：两个 Slider 各管一个参数。

本金、年化收益率分别拖动，余额曲线同步更新。
多参数的通用做法：所有滑块绑同一个回调，
回调里统一读取各滑块当前值，一把重算。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

plt.rcParams.update(
    {
        "font.sans-serif": ["PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC"],
        "axes.unicode_minus": False,
        "figure.dpi": 100,
    }
)

years = np.arange(0, 31)

fig, ax = plt.subplots(figsize=(8, 5))
fig.subplots_adjust(bottom=0.3)   # 底部要放两个滑块，多腾一点

(line,) = ax.plot(years, 10_000 * 1.05 ** years, lw=2)
ax.set_title("复利曲线：本金与收益率双滑块")
ax.set_xlabel("年数")
ax.set_ylabel("账户余额（元）")

deposit_ax = fig.add_axes((0.22, 0.15, 0.6, 0.03))
rate_ax = fig.add_axes((0.22, 0.08, 0.6, 0.03))
deposit_slider = Slider(deposit_ax, "本金(万)", 1, 50, valinit=10)
rate_slider = Slider(rate_ax, "年化收益率", 0.01, 0.20, valinit=0.05, valstep=0.005)


def refresh(_) -> None:
    """不管哪个滑块动了，都按两个滑块的当前值整条重算。"""
    balance = deposit_slider.val * 10_000 * (1 + rate_slider.val) ** years
    line.set_ydata(balance)
    fig.canvas.draw_idle()


deposit_slider.on_changed(refresh)
rate_slider.on_changed(refresh)

plt.show()
