"""动画演示一：FuncAnimation 的最小可用形态。

一条"全天气温"曲线随帧逐渐生长——
先摆一个空的折线 Artist，之后每来一帧，
update 就把曲线多推进一截。动画的底层逻辑：
不重画整图，只替换 Artist 携带的数据。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams.update(
    {
        "font.sans-serif": ["PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC"],
        "axes.unicode_minus": False,
        "figure.dpi": 100,
    }
)

# 每 10 分钟一个采样点，构造一天的气温曲线（午后最高）
hours = np.linspace(0, 24, 145)
temps = 16 + 8 * np.sin((hours - 8) / 24 * 2 * np.pi) + np.random.normal(0, 0.3, 145)

fig, ax = plt.subplots(figsize=(8, 4))
(line,) = ax.plot([], [], lw=2, color="orangered")   # 先挂上空 Artist
ax.set_xlim(0, 24)
ax.set_ylim(5, 30)
ax.set_title("全天气温：曲线随帧生长")
ax.set_xlabel("时刻（点）")
ax.set_ylabel("气温（℃）")


def update(frame: int):
    """第 frame 帧：曲线推进到第 frame 个采样点。"""
    line.set_data(hours[:frame], temps[:frame])
    return (line,)   # 告诉动画器这一帧动过哪些 Artist


# ani 必须留在变量里，否则可能被垃圾回收，动画会莫名停掉
ani = FuncAnimation(fig, update, frames=len(hours) + 1, interval=30, repeat=True)

plt.show()
