"""交互演示五：RadioButtons 在互斥选项间切换。

同一份 90 天营收数据，三种统计口径（原始 / 7 日均值 / 30 日均值）
只能选一个：勾到哪个，曲线立刻换一副面孔。
适合切换视图、口径、数据源这类"多选一"场景。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

plt.rcParams.update(
    {
        "font.sans-serif": ["PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC"],
        "axes.unicode_minus": False,
        "figure.dpi": 100,
    }
)

rng = np.random.default_rng(7)
days = np.arange(1, 91)
revenue = (
    5000
    + 35 * days
    + 800 * np.sin(days / 90 * 4 * np.pi)
    + rng.normal(0, 450, 90)
)


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    """滑动均值：宽度越大曲线越平滑（边缘为近似值）。"""
    return np.convolve(values, np.ones(width) / width, mode="same")


views = {
    "原始日营收": revenue,
    "7 日均值": moving_average(revenue, 7),
    "30 日均值": moving_average(revenue, 30),
}

fig, ax = plt.subplots(figsize=(8, 5))
fig.subplots_adjust(left=0.28)

(line,) = ax.plot(days, revenue, lw=2, color="royalblue")
ax.set_title("营收口径切换：原始 / 7 日 / 30 日")
ax.set_xlabel("天")
ax.set_ylabel("营收（元）")

radio_ax = fig.add_axes((0.05, 0.4, 0.18, 0.2))
radio = RadioButtons(radio_ax, tuple(views))


def switch(name: str) -> None:
    """切到哪个口径，就把曲线换成对应数据。"""
    line.set_ydata(views[name])
    fig.canvas.draw_idle()


radio.on_clicked(switch)

plt.show()
