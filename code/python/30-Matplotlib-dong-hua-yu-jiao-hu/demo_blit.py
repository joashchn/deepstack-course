"""动画演示三：blit + init_func 的性能组合。

blit=True 时画面里静止的部分只画一次，
每帧只重绘 update 返回的那几个 Artist。
配合 init_func 先布置好静态部分（坐标范围等），
数据点多的时候帧率提升肉眼可见。
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

# 一小时的室温基础曲线
minutes = np.linspace(0, 60, 300)
base = 22 + 2 * np.sin(minutes / 60 * 2 * np.pi)

fig, ax = plt.subplots(figsize=(8, 4))
(curve,) = ax.plot([], [], lw=2, color="teal")
(cursor,) = ax.plot([], [], "o", color="crimson", ms=9)   # 追在曲线末端的游标
ax.set_title("室温波动：blit 只重绘动过的部分")
ax.set_xlabel("分钟")
ax.set_ylabel("温度（℃）")


def init():
    """开播前布置静态部分，并返回参与 blit 的 Artist。"""
    ax.set_xlim(0, 60)
    ax.set_ylim(18, 27)
    return curve, cursor


def update(frame: int):
    """每帧整体波形漂移一点，游标咬住曲线右端。"""
    y = base + 0.4 * np.sin(minutes * 0.7 + frame * 0.08)
    curve.set_data(minutes, y)
    cursor.set_data([minutes[-1]], [y[-1]])
    return curve, cursor


ani = FuncAnimation(fig, update, frames=200, init_func=init, blit=True, interval=40)

plt.show()
