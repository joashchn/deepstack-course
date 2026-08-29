"""交互演示六：TextBox 接收精确的数值输入。

滑块适合"扫范围"，文本框适合"给精确值"——
输入每月存款额，回车后立刻重画 20 年储蓄曲线。
on_submit 拿到的是字符串，float() 转换失败就安静忽略，
用户敲错不该被一屏报错糊脸。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

plt.rcParams.update(
    {
        "font.sans-serif": ["PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC"],
        "axes.unicode_minus": False,
        "figure.dpi": 100,
    }
)

months = np.arange(0, 241)   # 20 年 = 240 个月
monthly_rate = 0.03 / 12     # 年化 3% 折成月利率


def balance(monthly: float) -> np.ndarray:
    """每月固定存一笔，按月复利滚存的总金额。"""
    return monthly * ((1 + monthly_rate) ** months - 1) / monthly_rate


fig, ax = plt.subplots(figsize=(8, 4))
fig.subplots_adjust(bottom=0.2)

(line,) = ax.plot(months / 12, balance(2000), lw=2)
ax.set_title("储蓄计划：输入月存款额后回车")
ax.set_xlabel("年数")
ax.set_ylabel("总额（元）")

box_ax = fig.add_axes((0.2, 0.06, 0.6, 0.06))
saving_box = TextBox(box_ax, "月存款", initial="2000")


def submit(text: str) -> None:
    """回车触发：字符串转数值，转不动就悄悄返回。"""
    try:
        amount = float(text)
    except ValueError:
        return
    line.set_ydata(balance(amount))
    fig.canvas.draw_idle()


saving_box.on_submit(submit)

plt.show()
