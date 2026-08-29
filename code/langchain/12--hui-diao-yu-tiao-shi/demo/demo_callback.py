"""拾光书屋 · 第 12 章 demo：回调与调试（单文件完整版）

前提（跑「在线模式」需要，离线可跳过）：
    1. 本机已安装并启动 Ollama（桌面端保持运行，或终端执行 `ollama serve`）
    2. 已拉取对话模型：ollama pull qwen3:4b
    3. 已安装依赖：pip install langchain langchain-ollama

运行：python demo_callback.py

离线降级：
    连不上 Ollama 时自动换成本文件内置的 StubBookAdvisor（假荐书官，
    BaseChatModel 子类，流式吐出固定荐语）。它和 ChatOllama 走的是同一套
    回调机制——on_chat_model_start / on_llm_new_token / on_llm_end 一个不少，
    业务代码一行不改。这正是回调的价值：你挂的是「事件」，不挑模型。

演示内容（对应课件小节）：
    1) 多回调并存：ConsoleTimingHandler（控制台计时）+ FileLogHandler（文件日志）
       一起挂到荐书链上流式跑（课件第 2、3 节）
    2) 工具钩子：on_tool_start / on_tool_end（课件第 4 节）
    3) 错误与回退：on_chain_error + with_fallbacks，失败→换备胎全程被观测
       （课件第 4 节，呼应 04 章的三张保险单）
    4) astream_events：事件流方式看链内部（课件第 5.2 节）
    5) set_debug：全局调试开关，截取开头几行展示（课件第 5.1 节）

运行产物：demo/callback_run.log —— FileLogHandler 写下的文件日志，跑完当场看。
"""

from __future__ import annotations

import asyncio
import io
import time
import urllib.request
from collections.abc import Iterator
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# ----------------------------- 可调参数 -----------------------------
OLLAMA_BASE_URL: str = "http://localhost:11434"
CHAT_MODEL: str = "qwen3:4b"
LOG_PATH: Path = Path(__file__).parent / "callback_run.log"  # 文件日志回调的输出
SET_DEBUG_PREVIEW_LINES: int = 14  # set_debug 演示截取的行数（全量输出太长）


# ----------------------------- 基础工具 -----------------------------
def ollama_alive(timeout_s: float = 2.0) -> bool:
    """探测本地 Ollama 服务是否在线。"""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout_s) as resp:
            return resp.status == 200
    except OSError:
        return False


class StubBookAdvisor(BaseChatModel):
    """离线降级用的假荐书官：固定文案、按块吐字、假装思考。

    继承 BaseChatModel 意味着它享受和 ChatOllama 完全相同的事件机制——
    框架不关心模型背后是本地推理还是写死的字符串，钩子照发不误。
    """

    reply: str = "《夜航船》：夜航人的掌灯，通勤路上随手翻几页最合适。"
    chunk_size: int = 6       # 每次吐几个字（模拟 token 粒度）
    delay_s: float = 0.05     # 每块之间的停顿（模拟生成耗时，让计时回调有数可报）

    @property
    def _llm_type(self) -> str:
        return "stub-book-advisor"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """非流式调用：一口气返回整句荐语。"""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """流式调用：按块吐字，并手动上报 token 事件。

        注意 on_llm_new_token 要由模型实现自己上报（真模型也是这么做的），
        这正是流式时回调能逐块听见的源头。
        """
        for i in range(0, len(self.reply), self.chunk_size):
            text = self.reply[i : i + self.chunk_size]
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
            if run_manager is not None:
                run_manager.on_llm_new_token(text, chunk=chunk)
            time.sleep(self.delay_s)
            yield chunk


# ----------------------------- 回调一：控制台计时 -----------------------------
def run_name(serialized: dict[str, Any] | None, kwargs: dict[str, Any]) -> str:
    """解析组件名：kwargs 里的 name 最全（serialized 为 None 时也有），serialized 兜底。"""
    return str(kwargs.get("name") or (serialized or {}).get("name") or "Runnable")


class ConsoleTimingHandler(BaseCallbackHandler):
    """控制台计时回调：每个 run 开始掐表，结束报耗时，流式时顺便数 token。"""

    def __init__(self) -> None:
        # run_id → 开始时刻；run_id → 收到的 token 块数
        self._starts: dict[UUID, float] = {}
        self._tokens: dict[UUID, int] = {}

    def _tick(self, run_id: UUID) -> float:
        self._starts[run_id] = time.perf_counter()
        return self._starts[run_id]

    def _tock(self, run_id: UUID) -> float:
        return time.perf_counter() - self._starts.pop(run_id, time.perf_counter())

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._tick(run_id)
        print(f"[计时] ▶ {run_name(serialized, kwargs)} 开工")

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        print(f"[计时] ■ 收工，耗时 {self._tock(run_id):.2f}s")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._starts.pop(run_id, None)  # 表要收回来，哪怕这一环翻车了
        print(f"[计时] ✗ 翻车：{error!r}")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._tick(run_id)
        self._tokens[run_id] = 0
        print(f"[计时] ▶ {run_name(serialized, kwargs)} 开工（进来 {len(messages[0])} 条消息）")

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # 流式结束帧会补一个空 token（chunk_position="last"），计数时滤掉
        if token:
            self._tokens[run_id] = self._tokens.get(run_id, 0) + 1

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        n = self._tokens.pop(run_id, 0)
        print(f"[计时] ■ 模型收工，耗时 {self._tock(run_id):.2f}s，吐了 {n} 个文本块")

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._tick(run_id)
        print(f"[计时] ▶ 工具 {run_name(serialized, kwargs)} 被调用：{input_str}")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        print(f"[计时] ■ 工具返回，耗时 {self._tock(run_id):.2f}s")


# ----------------------------- 回调二：文件日志 -----------------------------
class FileLogHandler(BaseCallbackHandler):
    """文件日志回调：模型的输入输出、工具调用、各种错误，追加写进日志文件。

    和 ConsoleTimingHandler 互不干扰——多回调并存，一个管屏幕一个管档案。
    """

    def __init__(self, path: Path = LOG_PATH) -> None:
        self._path = path
        # demo 要能反复跑，开跑前清空旧日志
        path.write_text("", encoding="utf-8")

    def _write(self, text: str) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(text)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        lines = "\n".join(f"    [{m.type}] {m.content}" for m in messages[0])
        self._write(f"[模型开始] {run_name(serialized, kwargs)}（run {str(run_id)[:8]}）\n{lines}\n")

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        text = response.generations[0][0].text  # 双层列表：取第一个请求的第一个生成
        self._write(f"[模型结束] 生成：{text}\n\n")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(f"[模型出错] {error!r}\n\n")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(f"[链出错]   {error!r}\n")

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = run_name(serialized, kwargs)
        self._write(f"[工具开始] {name}({input_str})\n")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(f"[工具结束] 返回：{output}\n")


# ----------------------------- 演示用的链与工具 -----------------------------
def build_chain(model: BaseChatModel):
    """03 章的老朋友：提示词 | 模型 | 解析器。"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是拾光书屋的荐书官，回复不超过两句话。"),
        ("human", "向{reader}推荐一本《{book}》"),
    ])
    return prompt | model | StrOutputParser()


@tool
def check_open(hour: int) -> str:
    """查询拾光书屋在指定时刻是否营业。"""
    return "营业中，欢迎来店" if 10 <= hour < 22 else "已打烊，明早 10 点见"


# ----------------------------- 演示步骤 -----------------------------
def demo_multi_callbacks(chain, file_log: FileLogHandler) -> None:
    """课件第 2、3 节：计时 + 文件日志两个回调并存，挂在链上流式跑。"""
    print("=" * 62)
    print("== 1) 多回调并存：控制台计时 + 文件日志，挂同一条链 ==")
    print("调用方式：chain.stream(..., config={'callbacks': [计时, 日志]})")
    print("-" * 62)
    timing = ConsoleTimingHandler()
    parts: list[str] = []
    for chunk in chain.stream(
        {"reader": "通勤族", "book": "夜航船"},
        config={"callbacks": [timing, file_log]},
    ):
        parts.append(chunk)
    print(f"链的最终输出：{''.join(parts)}")
    print()


def demo_tool_hooks(file_log: FileLogHandler) -> None:
    """课件第 4 节：工具钩子——Agent 时代审计「模型到底调了什么」靠它。"""
    print("=" * 62)
    print("== 2) 工具钩子：on_tool_start / on_tool_end ==")
    timing = ConsoleTimingHandler()
    answer = check_open.invoke({"hour": 15}, config={"callbacks": [timing, file_log]})
    print(f"工具返回：{answer}")
    print("(同一份日志也写进了 callback_run.log)")
    print()


def demo_error_and_fallback(file_log: FileLogHandler) -> None:
    """课件第 4 节 + 呼应 04 章：主链抛错 → with_fallbacks 换备胎，全程被观测。"""
    print("=" * 62)
    print("== 3) 错误与回退：on_chain_error + with_fallbacks（04 章的保险单）==")

    def look_up(book: dict[str, str]) -> str:
        if book["book"] == "绝版书":
            raise ValueError("绝版书查不到库存")
        return f"《{book['book']}》有货"

    finder = RunnableLambda(look_up).with_fallbacks(
        [RunnableLambda(lambda b: "《绝版书》已绝版，可到前台登记求书")]
    )
    timing = ConsoleTimingHandler()
    result = finder.invoke({"book": "绝版书"}, config={"callbacks": [timing, file_log]})
    print(f"最终拿到：{result}")
    print("（观察上面的输出：主链翻车 → 错误钩子响 → 备胎接管 → 备胎收工）")
    print()


async def demo_astream_events(chain) -> None:
    """课件第 5.2 节：不写 handler，直接把链的执行当事件流消费。"""
    print("=" * 62)
    print("== 4) astream_events：事件流方式看链内部（version='v2'）==")
    async for ev in chain.astream_events(
        {"reader": "夜猫子", "book": "山月记"}, version="v2"
    ):
        data: dict[str, Any] = ev.get("data", {})
        info = ""
        if "output" in data:  # end 类事件：优先看产出
            info = f"输出 {str(data['output'])[:36]}"
        elif "chunk" in data:  # stream 类事件：看流式块（消息块取 content）
            chunk = data["chunk"]
            text = getattr(chunk, "content", chunk)
            info = f"块 {str(text)[:20]}"
        elif "input" in data:  # start 类事件：看进料
            info = f"输入 {str(data['input'])[:36]}"
        print(f"  {ev['event']:<22} {ev['name']:<20} {info}")
    print()


def demo_set_debug(chain) -> None:
    """课件第 5.1 节：全局调试开关。全量输出很长，这里截取开头几行。"""
    print("=" * 62)
    print("== 5) set_debug：一行代码打开全量日志（截选）==")
    from langchain_core.globals import set_debug

    buf = io.StringIO()
    set_debug(True)
    try:
        with redirect_stdout(buf):  # 全量输出太长，接住再截选
            chain.invoke({"reader": "会员", "book": "小王子"})
    finally:
        set_debug(False)  # 看完就关，别一直刷屏

    lines = buf.getvalue().splitlines()
    for line in lines[:SET_DEBUG_PREVIEW_LINES]:
        print(f"  {line}")
    print(f"  ……（完整输出共 {len(lines)} 行，方括号里的路径就是执行树）")
    print()


def show_log_file() -> None:
    """把文件日志回调写下的档案摊开看。"""
    print("=" * 62)
    print(f"== 尾声：FileLogHandler 写下的 {LOG_PATH.name} ==")
    print("-" * 62)
    print(LOG_PATH.read_text(encoding="utf-8"), end="")
    print("-" * 62)


# ----------------------------- 主流程 -----------------------------
def main() -> None:
    print("拾光书屋 · 第 12 章 demo：回调与调试")
    print("=" * 62)

    online = ollama_alive()
    if online:
        print(f"[在线模式] 检测到 Ollama（{OLLAMA_BASE_URL}），使用 {CHAT_MODEL}。")
        # qwen3 默认在正文里输出 <think> 标签，reasoning=False 直接关掉思考
        model: BaseChatModel = ChatOllama(model=CHAT_MODEL, temperature=0.7, reasoning=False)
    else:
        print("[离线模式] 未检测到 Ollama，自动降级为内置假荐书官 StubBookAdvisor。")
        print("           它继承 BaseChatModel，钩子触发和真模型完全一致——")
        print("           回调挂在「事件」上，不挑模型。启动 Ollama 后重跑即用真模型。")
        model = StubBookAdvisor()

    chain = build_chain(model)
    file_log = FileLogHandler()  # 构造时清空旧日志，之后全程复用这一个

    demo_multi_callbacks(chain, file_log)
    demo_tool_hooks(file_log)
    demo_error_and_fallback(file_log)
    asyncio.run(demo_astream_events(chain))
    demo_set_debug(chain)
    show_log_file()


if __name__ == "__main__":
    main()
