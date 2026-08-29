"""交互演示一：单个 Slider 拖动调整连续参数。

滑块控制年化收益率，复利曲线实时重算。
交互三步套路的样板：
腾地方（fig.add_axes）→ 绑回调（on_changed）→
回调里改 Artist 并 draw_idle。
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
principal = 10_000   # 本金 1 万元

fig, ax = plt.subplots(figsize=(8, 4))
fig.subplots_adjust(bottom=0.25)   # 展格下方腾出滑块的位置

(line,) = ax.plot(years, principal * 1.05 ** years, lw=2)
ax.set_title("复利曲线：拖动滑块调整年化收益率")
ax.set_xlabel("年数")
ax.set_ylabel("账户余额（元）")

# 滑块的落脚处：四个数是 0~1 的画布比例坐标，与数据坐标无关
slider_ax = fig.add_axes((0.2, 0.1, 0.6, 0.03))
rate_slider = Slider(
    slider_ax, "年化收益率", 0.01, 0.20, valinit=0.05, valstep=0.005
)


def refresh(rate: float) -> None:
    """回调收到滑块当前值，重算曲线。"""
    line.set_ydata(principal * (1 + rate) ** years)
    fig.canvas.draw_idle()   # "有空再重绘"，比强行重绘顺滑


rate_slider.on_changed(refresh)

plt.show()
