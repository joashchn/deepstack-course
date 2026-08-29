"""第 03 章 demo：LCEL 与 Runnable。

场景：拾光书屋阅读社区的「读者请求流水线」。
全程纯数据变换（RunnableLambda / RunnablePassthrough / RunnableParallel），
不启动 Ollama 也能完整运行；文末附一段带模型的完整链，默认关闭。

运行方式：
    python demo_lcel.py                # 纯数据演示（无需模型）
    USE_MODEL=1 python demo_lcel.py    # 追加模型演示（需本地 Ollama + qwen3:4b）
"""

import os

from pydantic import BaseModel

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
)

# 是否运行「带模型」的演示段落
USE_MODEL: bool = os.environ.get("USE_MODEL", "") == "1"


def banner(title: str) -> None:
    """打印分节标题，方便在终端里对照课件阅读。"""
    print(f"\n{'=' * 56}\n{title}\n{'=' * 56}")


# ----------------------------------------------------------------------
# 1. 管道的两种写法：| 运算符 与 .pipe()
#    场景：读者留言清洗 —— 去首尾空白 -> 压缩中间空格 -> 盖社区印章
# ----------------------------------------------------------------------
def strip_text(text: str) -> str:
    return text.strip()


def squeeze_spaces(text: str) -> str:
    return " ".join(text.split())


def stamp(text: str) -> str:
    return f"【拾光书屋】{text}"


def demo_pipe() -> None:
    banner("1. 管道：| 运算符 与 .pipe() 完全等价")

    via_or = RunnableLambda(strip_text) | RunnableLambda(squeeze_spaces) | RunnableLambda(stamp)
    via_dot = (
        RunnableLambda(strip_text)
        .pipe(RunnableLambda(squeeze_spaces))
        .pipe(RunnableLambda(stamp))
    )

    msg = "  想找 一本   关于 星际旅行 的 书  "
    print("| 运算符 :", via_or.invoke(msg))
    print(".pipe() :", via_dot.invoke(msg))

    # .pipe() 还能直接接收普通函数（自动包成 RunnableLambda）
    auto_wrap = RunnableLambda(strip_text).pipe(squeeze_spaces)
    print("自动包装 :", auto_wrap.invoke("   拾光   书屋   "))


# ----------------------------------------------------------------------
# 2. 统一接口三兄弟：invoke / stream / batch
# ----------------------------------------------------------------------
def demo_three_brothers() -> None:
    banner("2. invoke / stream / batch 三兄弟")

    price_of = RunnableLambda(lambda b: f"《{b['title']}》定价 {b['price']} 元")

    # invoke：一次一个
    print("invoke :", price_of.invoke({"title": "小王子", "price": 39}))

    # stream：逐块产出（纯数据链通常只有一块）
    chunks = [c for c in price_of.stream({"title": "夜航", "price": 28})]
    print("stream :", chunks)

    # batch：一批进去，一批出来，顺序与输入一一对应
    cart = [
        {"title": "小王子", "price": 39},
        {"title": "夜航", "price": 28},
        {"title": "人类群星闪耀时", "price": 45},
    ]
    for i, r in enumerate(price_of.batch(cart), start=1):
        print(f"batch 第{i}条 :", r)


# ----------------------------------------------------------------------
# 3. RunnableLambda：普通函数变标准件（含类型标注对 input_schema 的影响）
# ----------------------------------------------------------------------
class Book(BaseModel):
    """带类型标注的输入模型，input_schema 能反映出它的形状。"""

    title: str
    pages: int


def tag_typed(b: Book) -> str:
    return f"{b.title}（{'长篇' if b.pages > 200 else '短篇'}）"


def demo_lambda() -> None:
    banner("3. RunnableLambda：普通函数上岗")

    def highlight(text: str) -> str:
        """给书名加高亮星标。"""
        return text.replace("小王子", "★小王子★")

    embellish = RunnableLambda(highlight)
    print("invoke :", embellish.invoke("本月共读书目是《小王子》"))

    # 函数只吃「一个」输入：多参数函数就包一层（输入通常用 dict）
    def tag_book(title: str, pages: int) -> str:
        return f"{title}（{'长篇' if pages > 200 else '短篇'}）"

    make_tag = RunnableLambda(lambda p: tag_book(p["title"], p["pages"]))
    print("包一层 :", make_tag.invoke({"title": "夜航", "pages": 120}))

    # 输入用 pydantic 模型标注后，input_schema 能反映真实的输入形状
    typed = RunnableLambda(tag_typed)
    print("schema :", list(typed.input_schema.model_json_schema()["properties"]))
    print("invoke :", typed.invoke(Book(title="人类群星闪耀时", pages=352)))


# ----------------------------------------------------------------------
# 4. RunnablePassthrough：直通 + assign 追加字段
# ----------------------------------------------------------------------
def demo_passthrough() -> None:
    banner("4. RunnablePassthrough：直通与 assign")

    # 直通：进什么，出什么
    print("直通    :", RunnablePassthrough().invoke("拾光书屋"))

    # assign：原字段保留，新字段并行追加（值函数拿到的是整个输入 dict）
    contextualize = RunnablePassthrough.assign(
        community=lambda _: "拾光书屋",
        member_level=lambda q: "黄金会员" if len(q["query"]) > 10 else "普通会员",
    )
    print("assign  :", contextualize.invoke({"query": "推荐几本关于星际旅行的书"}))


# ----------------------------------------------------------------------
# 5. RunnableParallel：同一份输入，多路并发
# ----------------------------------------------------------------------
def demo_parallel() -> None:
    banner("5. RunnableParallel：一变多的分身术")

    stats = RunnableParallel(
        title_len=lambda b: len(b["title"]),
        is_long=lambda b: b["pages"] > 300,
    )
    print("并行统计 :", stats.invoke({"title": "时间简史", "pages": 268}))

    # 直接把 dict 交给 | ，会被自动转成 RunnableParallel
    auto_parallel = RunnablePassthrough() | {
        "upper": lambda q: q["query"].upper(),
        "chars": lambda q: len(q["query"]),
    }
    print("自动并行 :", auto_parallel.invoke({"query": "star"}))


# ----------------------------------------------------------------------
# 6. 条件路由：RunnableBranch（声明式）与逻辑分支（函数式）
# ----------------------------------------------------------------------
def build_branch_desk() -> RunnableBranch:
    """写法一：RunnableBranch 声明式路由表。"""
    order_chain = RunnableLambda(lambda q: f"[订单台] 查询：{q['question']}")
    book_chain = RunnableLambda(lambda q: f"[荐书台] 荐书：{q['question']}")
    chat_chain = RunnableLambda(lambda q: f"[前台] 闲聊：{q['question']}")

    return RunnableBranch(
        (lambda q: "订单" in q["question"], order_chain),
        (lambda q: any(w in q["question"] for w in ("推荐", "找书", "书单")), book_chain),
        chat_chain,  # 兜底分支，不能省
    )


def triage(q: dict) -> str:
    """写法二：逻辑分支，判别 + 选择 + 执行收进一个函数。"""
    text = q["question"]
    if "订单" in text:
        return f"[订单台] 查询：{text}"
    if any(w in text for w in ("推荐", "找书", "书单")):
        return f"[荐书台] 荐书：{text}"
    return f"[前台] 闲聊：{text}"


def demo_routing() -> None:
    banner("6. 条件路由：RunnableBranch vs 逻辑分支")

    desk = build_branch_desk()
    logic_desk = RunnableLambda(triage)

    questions = [
        {"question": "我的订单到哪了"},
        {"question": "推荐几本科幻小说"},
        {"question": "今天天气不错"},
    ]
    for q in questions:
        print(f"Branch 路由 : {desk.invoke(q)}")
        print(f"逻辑路由   : {logic_desk.invoke(q)}")


# ----------------------------------------------------------------------
# 7. 链的嵌套组合：拼好的链继续当零件
# ----------------------------------------------------------------------
def demo_nesting() -> None:
    banner("7. 嵌套组合：RunnableSequence 与 | 混用")

    clean = RunnableLambda(lambda q: {**q, "query": q["query"].strip()})
    enrich = RunnablePassthrough.assign(scope=lambda _: "科幻")
    fanout = RunnableParallel(
        recommend=lambda q: f"荐书结果：{q['query']}",
        hot=lambda _: "本周热门：《三体》",
    )

    pipeline_a = clean | enrich | fanout               # | 拼装
    pipeline_b = RunnableSequence(clean, enrich, fanout)  # 显式步骤列表

    print("| 拼装      :", pipeline_a.invoke({"query": "  星际旅行  "}))
    print("Sequence   :", pipeline_b.invoke({"query": "  星际旅行  "}))

    # 拼好的 pipeline_a 本身是 Runnable，可以继续往后接
    final = pipeline_a | RunnableLambda(lambda r: f"一条通知：{r['recommend']}（来自 {r['hot']}）")
    print("继续拼装   :", final.invoke({"query": "  星际旅行  "}))


# ----------------------------------------------------------------------
# 8. Runnable 统一了什么：模板 / 模型 / 解析器 / 各种 Runnable 全是一家人
# ----------------------------------------------------------------------
def demo_unified() -> None:
    banner("8. Runnable 统一了什么")

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable

    members: list[tuple[str, object]] = [
        ("提示词模板", ChatPromptTemplate.from_template("推荐《{book}》")),
        ("字符串解析器", StrOutputParser()),
        ("RunnableLambda", RunnableLambda(lambda x: x)),
        ("RunnablePassthrough", RunnablePassthrough()),
        ("RunnableParallel", RunnableParallel(a=lambda x: x)),
        ("RunnableSequence", RunnableLambda(lambda x: x) | RunnableLambda(lambda x: x)),
    ]
    for name, m in members:
        print(f"{name:20s} isinstance(m, Runnable) -> {isinstance(m, Runnable)}")


# ----------------------------------------------------------------------
# 9.（可选）带模型的完整链：模板 -> 模型 -> 解析器 -> 后处理
# ----------------------------------------------------------------------
def demo_model_chain() -> None:
    banner("9. [可选] 带模型的完整链（需本地 Ollama 运行 qwen3:4b）")

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama

    model = ChatOllama(model="qwen3:4b", temperature=0.7)
    prompt = ChatPromptTemplate.from_template("用一句话向「拾光书屋」的读者推荐《{book}》")
    highlight = RunnableLambda(lambda s: s.replace("拾光书屋", "★拾光书屋★"))

    chain = prompt | model | StrOutputParser() | highlight
    print(chain.invoke({"book": "小王子"}))


def main() -> None:
    demo_pipe()
    demo_three_brothers()
    demo_lambda()
    demo_passthrough()
    demo_parallel()
    demo_routing()
    demo_nesting()
    demo_unified()

    if USE_MODEL:
        try:
            demo_model_chain()
        except Exception as exc:  # 模型没起来等情况，不影响前面的演示结论
            print(f"模型演示跳过：{type(exc).__name__}: {exc}")
    else:
        print("\n提示：设 USE_MODEL=1 可追加「带模型」的演示段落。")


if __name__ == "__main__":
    main()
