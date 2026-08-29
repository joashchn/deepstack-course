"""拾光书屋 · 第 11 章 demo：Agent（create_agent + 书屋工具，单文件版）

前提（跑「在线模式」缺一不可，运行前请自查）：
    1. 本机已安装并启动 Ollama（桌面端保持运行，或终端执行 `ollama serve`）
    2. 已拉取本章用到的对话模型：
        ollama pull qwen3:4b
    3. 已安装依赖：
        pip install langchain langchain-ollama

运行：python demo_agent.py

离线降级：
    连不上 Ollama 时自动进入「离线模式」：工具是本地纯函数、agent 构造也不需要
    联网，所以照样真实构造；但 agent 循环需要模型，改为打印「工具定义 + agent
    构造代码 + 在线时的消息流转示意」。启动 Ollama 后重跑即可看到完整效果。

演示内容（对应课件小节）：
    1) 四个本地纯函数工具：search_books / get_store_info / calculate_late_fee /
       get_member_events（课件第 4 节）
    2) create_agent 组装书屋小助手，一个「三连问」跑完整决策路径（课件第 2、3 节）
    3) 工具设计细节：描述对照 + 错误返回字符串而非抛异常（课件第 5 节）
"""

from __future__ import annotations

import json
import urllib.request

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph

# ----------------------------- 可调参数 -----------------------------
OLLAMA_BASE_URL: str = "http://localhost:11434"
CHAT_MODEL: str = "qwen3:4b"

# ----------------------------- 书屋「数据库」 -----------------------------
# 全部硬编码在文件里，工具都是本地纯函数，没有任何外部依赖。
BOOKS: dict[str, dict[str, str | int]] = {
    "夜航船": {"category": "古籍", "stock": 3, "note": "张岱的类书式笔记，条目短小，适合通勤"},
    "山月记": {"category": "小说", "stock": 0, "note": "中岛敦短篇，一个晚上读完"},
    "小王子": {"category": "童话", "stock": 5, "note": "亲子共读常客，周日故事会主角"},
    "城南旧事": {"category": "小说", "stock": 2, "note": "林海音自传体小说，文字干净克制"},
}

WEEKLY_EVENTS: dict[str, str] = {
    "周五": "19:00 读书会「短篇之夜」，共读《山月记》与《变形记》，限 12 人，免费",
    "周日": "15:00 儿童故事会（儿童区），带孩子直接来即可，无需报名",
}

STORE_INFO: str = (
    "营业时间：每天 10:00-22:00（周一不闭店）；"
    "借阅规则：会员一次最多借 5 本，借期 14 天，可续借一次；"
    "滞纳金：逾期每本书每天 0.5 元。"
)

SYSTEM_PROMPT: str = (
    "你是拾光书屋的店员。回答读者问题需要店内数据时必须调用工具，不要编造；"
    "读者问了多个问题时，依次回答，不要漏项。"
)

# ----------------------------- 工具定义 -----------------------------
# 写法与第 10 章一致：@tool + 类型标注 + docstring。
# 区别在于：第 10 章要自己写 while 循环消费这些工具，本章交给 create_agent。


@tool
def search_books(keyword: str) -> str:
    """按书名、类别或主题关键词搜索拾光书屋的在库图书，返回书名、库存与一句推荐语。"""
    hits: list[str] = [
        name
        for name, info in BOOKS.items()
        if keyword in name
        or keyword in str(info["category"])
        or keyword in str(info["note"])
    ]
    if not hits:
        return f"没有找到与「{keyword}」相关的书，可以换个说法再问我。"
    return "\n".join(
        f"《{name}》（{BOOKS[name]['category']}，库存 {BOOKS[name]['stock']} 本）：{BOOKS[name]['note']}"
        for name in hits
    )


@tool
def get_store_info() -> str:
    """查询拾光书屋的营业时间、借阅规则与滞纳金标准。"""
    return STORE_INFO


@tool
def calculate_late_fee(overdue_days: int, book_count: int) -> str:
    """计算会员逾期还书需要缴纳的滞纳金。书屋规定：逾期每本书每天收取 0.5 元。

    Args:
        overdue_days: 逾期天数
        book_count: 逾期未还的书籍数量
    """
    # 「参数合法但业务不成立」的情况：schema 只能保证是 int，保证不了非负——
    # 这种错误返回字符串（模型看到能自己纠正重试），千万别 raise（会击穿整个 agent）
    if overdue_days < 0 or book_count < 0:
        return "参数不合法：逾期天数和书籍数量都必须是非负整数，请核对后重试。"
    fee = overdue_days * book_count * 0.5
    return f"滞纳金共 {fee:.1f} 元（{book_count} 本 × {overdue_days} 天 × 0.5 元）。"


@tool
def get_member_events(weekday: str) -> str:
    """查询拾光书屋本周某天的会员活动（读书会、故事会等）。

    Args:
        weekday: 星期几，如「周五」「周日」
    """
    if weekday not in WEEKLY_EVENTS:
        return f"本周没有安排在「{weekday}」的活动。本周有活动的日子：{'、'.join(WEEKLY_EVENTS)}。"
    return f"本周{weekday}：{WEEKLY_EVENTS[weekday]}。"


# ----------------------------- 基础工具 -----------------------------
def ollama_alive(timeout_s: float = 2.0) -> bool:
    """探测本地 Ollama 服务是否在线。"""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout_s) as resp:
            return resp.status == 200
    except OSError:
        return False


def build_tools() -> list[BaseTool]:
    """把四个工具收进列表，交给 create_agent。"""
    return [search_books, get_store_info, calculate_late_fee, get_member_events]


def build_agent(tools: list[BaseTool]) -> CompiledStateGraph:
    """create_agent 三件套：model + tools + system_prompt，循环框架包办。"""
    # qwen3 默认会在正文里输出 <think> 标签，reasoning=False 直接关掉思考
    model = ChatOllama(model=CHAT_MODEL, temperature=0, reasoning=False)
    return create_agent(model, tools, system_prompt=SYSTEM_PROMPT)


def print_trace(messages: list[BaseMessage]) -> None:
    """把一次 agent 运行的完整决策路径打印出来（课件第 3 节同款）。"""
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for call in m.tool_calls:
                print(f"  [决策] 调用 {call['name']}({call['args']})")
        elif isinstance(m, ToolMessage):
            text = m.content.replace("\n", " / ")
            print(f"  [观察] {m.name} 返回：{text[:60]}…")
        else:
            text = m.content.replace("\n", " ")
            print(f"  [{type(m).__name__}] {text[:70]}")


# ----------------------------- 演示步骤 -----------------------------
def demo_tools_overview(tools: list[BaseTool]) -> None:
    """课件第 4、5 节：先看清挂在墙上的四张「工具卡」。

    模型选工具的唯一依据就是这里的 name / description / 参数 schema——
    这就是课件反复强调「描述要写清楚」的原因。
    """
    print("=" * 62)
    print("== 1) 工具挂墙：四个本地纯函数工具 ==")
    for t in tools:
        print(f"  ● {t.name}")
        print(f"    描述：{t.description}")
        if t.args:
            print(f"    参数：{json.dumps(t.args, ensure_ascii=False)}")
        else:
            print("    参数：（无）")
    print()


def demo_agent_run(agent: CompiledStateGraph, question: str) -> None:
    """课件第 2、3、4 节：一个「三连问」跑完整决策路径。"""
    print("=" * 62)
    print("== 2) create_agent 实战：三连问，多工具协作 ==")
    print(f"问题：{question}")
    print("--- 决策路径（result['messages'] 的内容）---")
    result: dict = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print_trace(result["messages"])
    print("--- 最终答案（result['messages'][-1].content）---")
    print(result["messages"][-1].content)
    print()


def demo_agent_offline() -> None:
    """离线降级：agent 已真实构造（构造不联网），但循环需要模型，改为展示。"""
    print("=" * 62)
    print("== 2) create_agent 组装（离线：只构造、不调用）==")
    print("[说明] 未检测到 Ollama，agent 循环需要真实模型，本轮不执行 invoke。")
    print("       工具是本地纯函数，agent 构造也不需要联网——所以上面打印的")
    print("       工具定义是真实对象，下面的构造代码也已真实执行。")
    print("-" * 62)
    print("等价的 agent 构造代码（本文件 build_agent() 的核心三行）：")
    print("  from langchain.agents import create_agent")
    print("  model = ChatOllama(model='qwen3:4b', temperature=0, reasoning=False)")
    print("  agent = create_agent(model, tools, system_prompt='你是拾光书屋的店员……')")
    print("-" * 62)
    print("在线时执行：agent.invoke({'messages': [{'role': 'user', 'content': '……'}]})")
    print("返回的 result['messages'] 是完整决策路径，形如：")
    print("  HumanMessage  （读者提问）")
    print("  AIMessage     （tool_calls: search_books —— 模型决策）")
    print("  ToolMessage   （工具返回 —— 框架自动执行并回填）")
    print("  AIMessage     （最终答案 —— 循环结束）")
    print("多工具任务时中间两棒会来回传多趟，直到模型不再发起工具调用。")
    print(f"启动 Ollama 并执行 `ollama pull {CHAT_MODEL}` 后重跑，即可看到真实运行。")
    print()


def demo_tool_design() -> None:
    """课件第 5 节：描述对照 + 错误返回字符串。全程本地，无需模型。"""
    print("=" * 62)
    print("== 3) 工具设计细节：描述、粒度、报错 ==")
    print("-- 描述对照（给模型看的说明书）--")
    print("  反例：search(data: str)「搜索」——模型不知道何时该用、参数填什么")
    print("  正例：search_books(keyword: str)")
    print(f"        「{search_books.description}」")
    print()
    print("-- 错误返回字符串（直接调用工具演示，不经过模型）--")
    print(f"  calculate_late_fee(overdue_days=-1, book_count=3)")
    print(f"    → {calculate_late_fee.invoke({'overdue_days': -1, 'book_count': 3})}")
    print(f"  get_member_events(weekday='周八')")
    print(f"    → {get_member_events.invoke({'weekday': '周八'})}")
    print()
    print("  说明：这两处都是「参数合法但业务不成立」的错误——返回描述性字符串，")
    print("  模型看到后能自己换参数重试或改推荐方案；如果改成 raise，LangChain 1.x")
    print("  里默认会直接击穿整个 agent，前面跑的几步全部白费（课件 5.3 实测）。")
    print()


# ----------------------------- 主流程 -----------------------------
def main() -> None:
    print("拾光书屋 · 第 11 章 demo：Agent（create_agent）")
    print("=" * 62)

    tools = build_tools()
    demo_tools_overview(tools)

    # agent 构造本身不需要联网；联网只在 invoke 阶段发生
    agent = build_agent(tools)

    if ollama_alive():
        print(f"[在线模式] 检测到 Ollama（{OLLAMA_BASE_URL}），使用真实模型 {CHAT_MODEL}。")
        print()
        demo_agent_run(
            agent,
            "《夜航船》还有吗？另外我上次借的 3 本书逾期了 5 天，"
            "要交多少滞纳金？对了，这周五晚上有读书会吗，几点开始？",
        )
    else:
        print(f"[离线模式] 未检测到 Ollama（{OLLAMA_BASE_URL}）。")
        print()
        demo_agent_offline()

    # 第 3 部分纯本地，在线/离线都跑
    demo_tool_design()


if __name__ == "__main__":
    main()
