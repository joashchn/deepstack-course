"""拾光书屋 · 第 09 章 demo：对话记忆（单文件完整版）

前提（跑「在线模式」缺一不可，运行前请自查）：
    1. 本机已安装并启动 Ollama（桌面端保持运行，或终端执行 `ollama serve`）
    2. 已拉取本章用到的对话模型：
        ollama pull qwen3:4b
    3. 已安装依赖：
        pip install langchain langchain-ollama

运行：python demo_memory.py

离线降级：
    历史管理逻辑（列表维护、多会话隔离、trim_messages、JSON 持久化）
    完全不需要模型；模型相关的两个环节——店员回复、对话摘要——
    离线时换成内置的「剧本回复」和「截断式假摘要」，流程照样走通。
    其中剧本回复会真的从历史里翻出会员号来回答——记忆机制本身被当场验证。
    启动 Ollama 后重跑即可看到真实模型版。

演示内容（对应课件小节）：
    1) 手动维护 messages：三轮对话，看「记忆」如何随列表增长（课件第 1 节）
    2) 多会话管理：session_id 隔离两位会员的记录本（课件第 2 节）
    3) trim_messages：按条数/按 token 开窗，include_system 与 start_on 对比（课件第 3 节）
    4) 摘要压缩：旧消息 → 便签 + 保留最近几条（课件第 4 节）
    5) JSON 持久化：原子写入 → 模拟重启读回 → 坏文件兜底（课件第 5 节）
    6) 第三方存储：升级信号 checklist 打印（课件第 6 节，概念）
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections.abc import Callable

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
    messages_from_dict,
    messages_to_dict,
    trim_messages,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------- 可调参数 -----------------------------
OLLAMA_BASE_URL: str = "http://localhost:11434"
CHAT_MODEL: str = "qwen3:4b"        # 对话模型
WINDOW_KEEP: int = 4                # 窗口保留的最近消息条数
SUMMARY_TRIGGER: int = 10           # 历史超过多少条触发摘要压缩
SUMMARY_KEEP_RECENT: int = 4        # 摘要压缩时保留的最近消息条数
HISTORY_FILE: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "history_member-2048.json"
)

SYSTEM_TEXT: str = "你是拾光书屋的店员，语气亲切，回答简洁。"

SUMMARY_TEMPLATE: str = (
    "你是拾光书屋的店长助理。请把下面这段店员与会员的对话压缩成一份摘要，"
    "务必保留：会员称呼、会员号、阅读偏好、借阅/预约状态、答应过会员的事。"
    "不超过 5 句话。\n\n【对话记录】\n{history}"
)

# ----------------------------- 基础工具 -----------------------------
def ollama_alive(timeout_s: float = 2.0) -> bool:
    """探测本地 Ollama 服务是否在线。"""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout_s) as resp:
            return resp.status == 200
    except OSError:
        return False


def build_reply_fn(online: bool) -> Callable[[list[AnyMessage]], str]:
    """造一支「店员之笔」：在线用真实模型，离线用剧本。

    统一约定：吃「到目前为止的完整消息列表」，吐一段字符串回复。
    """
    if online:
        from langchain_ollama import ChatOllama

        # qwen3 默认会在正文里输出 <think> 标签，reasoning=False 直接关掉思考
        model = ChatOllama(model=CHAT_MODEL, temperature=0, reasoning=False)

        def model_reply(messages: list[AnyMessage]) -> str:
            return model.invoke(messages).content

        return model_reply

    def scripted_reply(messages: list[AnyMessage]) -> str:
        """离线剧本：从历史里「翻记录」回答，当场验证记忆机制在干活。"""
        last: str = str(messages[-1].content)
        if "会员号" in last or "号码" in last:
            for m in messages:
                if isinstance(m, HumanMessage):
                    match = re.search(r"会员号\s*(\d+)", str(m.content))
                    if match:
                        return f"您的会员号是 {match.group(1)}。"
            return "抱歉，登记簿上还没有您的会员号。"
        if "到期" in last or "还书" in last:
            return "《山月记》的借期到下周三，看完记得按时归还哦。"
        if "续借" in last:
            return "可以续借一次，共延长 14 天，这就帮您登记。"
        if "推荐" in last or "书" in last:
            return "推荐《城南旧事》，安静克制，适合泡壶茶慢慢读。"
        return "好的，还有什么可以帮您？"

    return scripted_reply


def build_summarize_fn(online: bool) -> Callable[[str], str]:
    """造一台「摘要机」：在线用真实模型，离线用截断式假摘要。"""
    if online:
        from langchain_ollama import ChatOllama

        model = ChatOllama(model=CHAT_MODEL, temperature=0, reasoning=False)
        summarizer = (
            ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
            | model
            | StrOutputParser()
        )

        def real_summarize(history_text: str) -> str:
            return str(summarizer.invoke({"history": history_text}))

        return real_summarize

    def fake_summarize(history_text: str) -> str:
        """离线兜底：每条消息截前 14 字拼接——模拟「压缩」，演示结构变化。"""
        lines = [ln for ln in history_text.splitlines() if ln.strip()]
        head = "；".join(ln[:14] for ln in lines[:4])
        return f"（离线假摘要）此前对话要点：{head}……"

    return fake_summarize


# ----------------------------- 会话封装 -----------------------------
class ChatSession:
    """一位会员的对话记录本：messages 列表 + 一支「店员之笔」。

    这就是课件第 2 节说的「三行代码的自动记忆」——
    入账 → 调用 → 回账，没有更多魔法。
    """

    def __init__(self, reply_fn: Callable[[list[AnyMessage]], str]) -> None:
        self.messages: list[AnyMessage] = []
        self._reply_fn = reply_fn

    def chat(self, user_input: str) -> str:
        """一轮对话：入账 → 调用 → 回账。"""
        self.messages.append(HumanMessage(user_input))
        reply = self._reply_fn(self.messages)   # 把整本记录递给店员
        self.messages.append(AIMessage(reply))
        return reply


def make_history(rounds: int) -> list[AnyMessage]:
    """造一段多轮历史（trim / 摘要演示用，不依赖模型）。"""
    messages: list[AnyMessage] = [SystemMessage(SYSTEM_TEXT)]
    for i in range(rounds):
        messages.append(HumanMessage(f"第 {i + 1} 问：书屋这周有什么活动？"))
        messages.append(AIMessage(f"第 {i + 1} 答：周五晚有「短篇之夜」读书会。"))
    return messages


def brief(messages: list[AnyMessage]) -> str:
    """把消息列表压成一行摘要文本，方便控制台观察。"""
    return " → ".join(
        f"{type(m).__name__[0]}:{str(m.content)[:10]}" for m in messages
    )


# ----------------------------- 演示步骤 -----------------------------
def demo_manual(session: ChatSession) -> None:
    """课件第 1 节：记忆的真相——三轮对话看列表如何生长。"""
    print("=" * 62)
    print("== 1) 手动维护 messages：三轮对话，记忆随列表生长 ==")
    for user_input in (
        "你好，我是小雅，刚办了拾光书屋的年卡，会员号 2048",
        "我上周借了一本《山月记》",
        "那我的会员号你还记得吗？",
    ):
        reply = session.chat(user_input)
        print(f"小雅：{user_input}")
        print(f"店员：{reply}")
        print(f"     （记录本现有 {len(session.messages)} 条消息）\n")


def demo_sessions(reply_fn: Callable[[list[AnyMessage]], str]) -> None:
    """课件第 2 节：多会话隔离——两位会员的记录本各管各的。"""
    print("=" * 62)
    print("== 2) 多会话管理：session_id 隔离，记录本互不串台 ==")
    sessions: dict[str, ChatSession] = {
        "member-2048": ChatSession(reply_fn),   # 小雅
        "member-1024": ChatSession(reply_fn),   # 小周
    }

    xiaoya = sessions["member-2048"]
    zhou = sessions["member-1024"]

    xiaoya.chat("我叫小雅，会员号 2048，喜欢短篇小说")
    zhou.chat("我叫小周，第一次来，帮我推荐一本书")
    print("小雅（member-2048）：", xiaoya.chat("我的会员号是多少？"))
    print("     ↑ 答得上来——因为问的是小雅自己的记录本\n")
    print("小周（member-1024）：", zhou.chat("我叫什么来着？"))
    print("     ↑ 记录本里只有小周自己说过的话，两边互不干扰\n")


def demo_trim() -> None:
    """课件第 3 节：trim_messages 开窗——默认坑 vs 正确姿势。"""
    print("=" * 62)
    print("== 3) trim_messages：给记忆开一扇滑动的窗 ==")
    messages = make_history(6)  # system + 12 条 = 13 条
    print(f"造一段长历史（{len(messages)} 条）：{brief(messages)}\n")

    # 3.1 裸调用：两个坑一次暴露——system 被裁掉、AI 开头（非法历史）
    #     预算取 3，让尾部装出 [A, H, A] 的 AI 开头结果，坑看得最清楚
    naive = trim_messages(
        messages, max_tokens=3, token_counter=len, strategy="last"
    )
    print("-- 裸调用（max_tokens=3, token_counter=len）--")
    print(f"   结果 {len(naive)} 条：{brief(naive)}")
    first = naive[0]
    print(f"   坑 1：第一条是 {type(first).__name__}"
          f"{'（AI 开头，多数模型不收）' if isinstance(first, AIMessage) else ''}")
    print(f"   坑 2：system 还在吗？"
          f"{'在' if any(isinstance(m, SystemMessage) for m in naive) else '没了——店员人设被当旧账扔了'}\n")

    # 3.2 正确姿势：system 保留 + human 开头
    #     注意（实测）：token_counter=len 时 system 也占预算名额，
    #     start_on='human' 还会额外掐掉开头那条 AI——预算要开足
    proper = trim_messages(
        messages,
        max_tokens=WINDOW_KEEP + 2,   # system 占 1 + start_on 可能丢 1，都补上
        token_counter=len,
        strategy="last",
        include_system=True,
        start_on="human",
    )
    print(f"-- 正确姿势：include_system=True + start_on='human'（预算开到 {WINDOW_KEEP + 2}）--")
    print(f"   结果 {len(proper)} 条：{brief(proper)}")
    ok_system = any(isinstance(m, SystemMessage) for m in proper)
    ok_human = isinstance(proper[0], SystemMessage) and isinstance(proper[1], HumanMessage)
    print(f"   system 保留：{'✓' if ok_system else '✗'}；human 开头：{'✓' if ok_human else '✗'}")
    print(f"   （system 占预算、start_on 额外丢一条——想留 {WINDOW_KEEP} 条正文，")
    print("     预算得开到 5 以上）\n")

    # 3.3 换 token 计数器：approximate 按 token 预算裁
    approx = trim_messages(
        messages,
        max_tokens=120,
        token_counter="approximate",
        strategy="last",
        include_system=True,
        start_on="human",
    )
    print("-- token_counter='approximate'：按 token 预算裁（max_tokens=120）--")
    print(f"   结果 {len(approx)} 条：{brief(approx)}")
    print("   （token_counter=len 时 max_tokens 数的是条数；")
    print("     换成 approximate 才是真的按 token 估，热路径推荐）\n")

    # 3.4 trim 不动原件：返回新列表
    print(f"-- trim_messages 返回新列表，原件不动 --")
    print(f"   原历史仍为 {len(messages)} 条；想替换要用 messages[:] = proper")
    print()


def demo_summary(summarize_fn: Callable[[str], str], online: bool) -> None:
    """课件第 4 节：长程记忆——旧消息压成便签 + 保留最近几条。"""
    print("=" * 62)
    print("== 4) 摘要压缩：旧对话压成一张便签 ==")
    messages = make_history(7)  # system + 14 条 = 15 条 > SUMMARY_TRIGGER
    print(f"造一段超阈值历史（{len(messages)} 条 > {SUMMARY_TRIGGER} 条触发压缩）\n")

    system = messages[0] if isinstance(messages[0], SystemMessage) else None
    body = messages[1:] if system else messages
    old, recent = body[:-SUMMARY_KEEP_RECENT], body[-SUMMARY_KEEP_RECENT:]
    summary_text = summarize_fn(get_buffer_string(old))
    note = SystemMessage(f"【此前对话摘要】{summary_text}")
    compressed = ([system] if system else []) + [note] + recent

    print(f"-- 压缩前 {len(messages)} 条 → 压缩后 {len(compressed)} 条 --")
    print(f"   便签内容：{summary_text[:60]}…")
    print(f"   结构：system + 摘要便签 + 最近 {SUMMARY_KEEP_RECENT} 条原文")
    print(f"   压缩后：{brief(compressed)}\n")

    # 渐进式摘要：压缩后继续聊，再次超阈值时「旧便签 + 新挤出的消息」再压
    print("-- 渐进式摘要（复印件的复印件，失真会累积）--")
    print("   下次触发时，把『旧便签 + 新挤出的消息』一起喂给摘要机；")
    print("   关键事实（会员号等）建议写死在 system，别只指望摘要链。")
    print(f"   当前摘要来源：{'真实模型' if online else '离线假摘要（截断拼接）'}。\n")


def demo_persist() -> None:
    """课件第 5 节：JSON 持久化——原子写入、重启读回、坏文件兜底。"""
    print("=" * 62)
    print("== 5) JSON 持久化：打烊前把白板拍照存档 ==")
    messages: list[AnyMessage] = [
        SystemMessage(SYSTEM_TEXT),
        HumanMessage("我是小雅，会员号 2048，上周借了《山月记》"),
        AIMessage("已登记！《山月记》借期 14 天，下周三到期。"),
    ]

    # 5.1 原子写入：先写 .tmp，再 os.replace 整体换页
    tmp = f"{HISTORY_FILE}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(messages_to_dict(messages), f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())          # 确保真的落盘
    os.replace(tmp, HISTORY_FILE)     # 原子换页：要么旧文件要么新文件
    print(f"已原子写入 {len(messages)} 条消息 → {os.path.basename(HISTORY_FILE)}")
    print(f"   （.tmp 临时文件已随换页消失：{not os.path.exists(tmp)}）\n")

    # 5.2 模拟进程重启：内存清空，从文件读回
    restored = load_messages(HISTORY_FILE)
    print(f"-- 模拟重启：从磁盘读回 {len(restored)} 条 --")
    print(f"   第 1 条：{type(restored[0]).__name__}: {str(restored[0].content)[:20]}…")
    print(f"   第 2 条：{type(restored[1]).__name__}: {str(restored[1].content)[:20]}…")
    print("   接着聊：restored 就是新的 messages 列表，无缝续杯\n")

    # 5.3 坏文件兜底：半张脸的 JSON 不会放倒整个服务
    bad_file = HISTORY_FILE + ".bad"
    with open(bad_file, "w", encoding="utf-8") as f:
        f.write('{"type": "human", "data": {"content": "写到一半断电了……')  # 残缺 JSON
    salvaged = load_messages(bad_file)
    print(f"-- 坏文件兜底：残缺 JSON 读回 {len(salvaged)} 条（发新本子，不崩服务）--")
    os.remove(bad_file)
    print()


def load_messages(path: str) -> list[AnyMessage]:
    """读回历史：文件不存在发新本子，坏了报警并兜底（课件第 5 节签名）。"""
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return list(messages_from_dict(json.load(f)))
    except json.JSONDecodeError:
        print(f"[警告] {os.path.basename(path)} 已损坏，启用新记录本")
        return []


def demo_third_party() -> None:
    """课件第 6 节：第三方存储——升级信号 checklist（概念）。"""
    print("=" * 62)
    print("== 6) 第三方存储：什么时候把 JSON 文件升级成数据库 ==")
    signals: list[tuple[str, str]] = [
        ("多进程", "uvicorn 开了多个 worker，内存 dict 互不相通 → 会话间歇性失忆"),
        ("多机部署", "任何一台的本地文件都不是全局 → 记录本搬进总部档案室"),
        ("要审计", "聊天记录要留档、检索、给运营分析 → Postgres 的主场"),
        ("写冲突", "文件锁成了新瓶颈 → 该上数据库了"),
    ]
    for name, why in signals:
        print(f"   [{name}] {why}")
    print("\n   存储选型：Redis = 热会话（快，记得设 TTL）；")
    print("             Postgres = 长期留存与审计。")
    print("   存的东西不变（消息列表序列化结构），变的只是读和写两个函数。")
    print("   框架级方案叫 LangGraph checkpointer，第 11 章 Agent 再见。\n")


# ----------------------------- 主流程 -----------------------------
def main() -> None:
    print("拾光书屋 · 第 09 章 demo：对话记忆")
    print("=" * 62)

    online = ollama_alive()
    if online:
        print(f"[在线模式] 检测到 Ollama（{OLLAMA_BASE_URL}），店员回复与摘要用真实模型。")
    else:
        print("[离线模式] 未检测到 Ollama，店员回复换内置剧本、摘要换截断假摘要；")
        print(f"           历史管理逻辑全程照常。请执行 `ollama pull {CHAT_MODEL}` 后重跑看完整版。")
    print()

    reply_fn = build_reply_fn(online)
    summarize_fn = build_summarize_fn(online)

    # 1) 手动维护 messages（需要「店员之笔」，离线走剧本）
    session = ChatSession(reply_fn)
    demo_manual(session)

    # 2) 多会话隔离（需要「店员之笔」，离线走剧本）
    demo_sessions(reply_fn)

    # 3) trim_messages（纯历史管理，不需要模型）
    demo_trim()

    # 4) 摘要压缩（需要「摘要机」，离线走假摘要）
    demo_summary(summarize_fn, online)

    # 5) JSON 持久化（纯文件操作，不需要模型）
    demo_persist()

    # 6) 第三方存储（概念打印，不需要模型）
    demo_third_party()

    # 收尾：把本 demo 产生的会话文件留在 demo 目录，可自行翻看
    print(f"（本 demo 的会话文件保存在：{HISTORY_FILE}，可打开翻看 JSON 结构）")


if __name__ == "__main__":
    main()
