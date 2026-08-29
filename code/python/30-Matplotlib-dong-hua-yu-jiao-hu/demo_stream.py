"""动画演示二：frames 交给无限生成器，模拟实时数据流。

一个"永远吐数据"的温度传感器模拟器，
配上 deque(maxlen=N) 滚动窗口：
新数据从右侧涌入，旧数据滑出左侧视野，
横轴随之右移——监控大屏上的滚动效果。
"""

from collections import deque

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

WINDOW = 120   # 画面只保留最近 120 个采样点

fig, ax = plt.subplots(figsize=(8, 4))
temps = deque(maxlen=WINDOW)   # 容量一满自动挤掉最老的值
(line,) = ax.plot([], [], lw=2)
ax.set_xlim(0, WINDOW)
ax.set_ylim(15, 35)
ax.set_title("机房温度：实时滚动窗口")
ax.set_xlabel("采样序号")
ax.set_ylabel("温度（℃）")


def sensor():
    """无限生成器：模拟一秒一条的传感器读数。"""
    tick, drift = 0, 0.0
    while True:
        drift = np.clip(drift + np.random.normal(0, 0.05), -1.5, 1.5)
        yield tick, 24 + drift + np.random.normal(0, 0.4)
        tick += 1


def update(frame: tuple[int, float]):
    """每收到一条读数，就往窗口里追加并刷新曲线。"""
    tick, value = frame
    temps.append(value)
    line.set_data(range(len(temps)), temps)
    if tick >= WINDOW:   # 窗口满了，横轴跟着右移
        ax.set_xlim(tick - WINDOW + 1, tick + 1)


# 帧数事先未知：cache_frame_data 必须关掉，否则无限缓存吃内存
ani = FuncAnimation(fig, update, frames=sensor(), interval=50, cache_frame_data=False)

plt.show()
