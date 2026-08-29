"""事件循环调度实验：对应课件习题三

先自己预测输出顺序，再运行 python demo.py 对答案。
重点观察：stop() 之后，本轮已就绪的回调是否还会执行。
"""

import asyncio

loop = asyncio.new_event_loop()


def warmup():
    print(1)


def showcase():
    print(2)
    # 把 print(3) 排进下一轮——但下一轮还会来吗？
    loop.call_soon(lambda: print(3))
    loop.stop()  # 打出停止标记


def finale():
    print(4)


loop.call_soon(warmup)
loop.call_soon(showcase)
loop.call_soon(finale)

loop.run_forever()
print("循环已停止")

# 预期输出：
# 1
# 2
# 4
# 循环已停止
# （3 永远不会出现：它排在下一轮，而循环在 showcase 里就被叫停了；
#  但 showcase、finale 是本轮交接出来的回调，仍会执行完）
