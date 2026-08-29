"""交互演示三：Button 触发一次动作。

点一下"再模拟一次"，30 天客流重新随机生成。
两个配套知识点：
hovercolor 悬停变色提示可点；
数据量级变了要用 relim + autoscale_view 让坐标轴重新适应。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

plt.rcParams.update(
    {
        "font.sans-serif": ["PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC"],
        "axes.unicode_minus": False,
        "figure.dpi": 100,
    }
)

rng = np.random.default_rng()

fig, ax = plt.subplots(figsize=(8, 4))
fig.subplots_adjust(bottom=0.2)

days = np.arange(1, 31)
(line,) = ax.plot(days, 200 + rng.normal(0, 5, 30).cumsum(),
                  lw=2, marker="o", ms=3)
ax.set_title("门店客流：点按钮再模拟一次")
ax.set_xlabel("天")
ax.set_ylabel("客流（百人次）")

button_ax = fig.add_axes((0.38, 0.05, 0.24, 0.075))
resim = Button(button_ax, "再模拟一次", color="lightyellow", hovercolor="gold")


def rerun(event) -> None:
    """重新生成一份随机游走数据并刷新画面。"""
    line.set_ydata(200 + rng.normal(0, 5, 30).cumsum())
    ax.relim()            # 按新数据重新划定数据边界
    ax.autoscale_view()   # 坐标轴随之自适应
    fig.canvas.draw_idle()


resim.on_clicked(rerun)

plt.show()
