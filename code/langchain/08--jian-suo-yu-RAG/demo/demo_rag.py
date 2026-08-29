"""拾光书屋 · 第 08 章 demo：检索与 RAG（单文件完整版）

前提（跑「完整模式」缺一不可，运行前请自查）：
    1. 本机已安装并启动 Ollama（桌面端保持运行，或终端执行 `ollama serve`）
    2. 已拉取本章用到的两个模型：
        ollama pull qwen3:4b          # 对话模型
        ollama pull nomic-embed-text  # 嵌入模型
    3. 已安装依赖：
        pip install langchain langchain-ollama langchain-text-splitters numpy
       （numpy 是 InMemoryVectorStore 计算余弦相似度的依赖）

运行：python demo_rag.py

离线降级：
    连不上 Ollama 时自动进入「离线模式」：嵌入换成文件内置的确定性词袋向量，
    只演示「语料 → 切割 → 入库 → 检索参数 → 组装提示词」；需要模型的部分
    （RAG 问答、多查询改写）跳过并打印提示。启动 Ollama 后重跑即可看到完整效果。

演示内容（对应课件小节）：
    1) 自造书屋读书笔记语料 → 切割 → 嵌入 → 入库（InMemoryVectorStore）
    2) 检索参数：k / 分数阈值过滤 / MMR / metadata filter（课件第 2、3 节）
    3) LCEL 组装 RAG 链：RunnableParallel 取上下文+传问题 → prompt → model → parser（课件第 4、5 节）
    4) 来源溯源：答案连同被引用的文档一起返回（课件第 6 节）
    5) 多查询检索：一个问题改写成多种说法，分头检索、合并去重（课件第 7 节）
"""

from __future__ import annotations

import urllib.request
import zlib
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------- 可调参数 -----------------------------
OLLAMA_BASE_URL: str = "http://localhost:11434"
CHAT_MODEL: str = "qwen3:4b"          # 对话模型
EMBED_MODEL: str = "nomic-embed-text"  # 嵌入模型
TOP_K: int = 3                        # 每次检索取几条
# 分数阈值的刻度随嵌入模型而变：nomic-embed-text 的相关文本大致落在 0.4~0.7，
# 词袋兜底向量整体低一个量级——这正是课件强调的「先打印分数分布再定阈值」
SCORE_THRESHOLD_ONLINE: float = 0.4
SCORE_THRESHOLD_OFFLINE: float = 0.1
CHUNK_SIZE: int = 150                 # 切块大小（字符）
CHUNK_OVERLAP: int = 30               # 相邻块重叠（字符）

# ----------------------------- 自造语料 -----------------------------
# 拾光书屋的读书笔记 + 店内公告，全部硬编码在文件里，无外部文件依赖。
READING_NOTES: list[dict[str, str]] = [
    {
        "book": "夜航船",
        "category": "古籍",
        "text": (
            "《夜航船》是明代张岱编的一部类书式笔记，全书二十卷，从天文地理、"
            "三教九流到草木鸟兽、衣食器用，无所不包。每条都很短，考据有趣，"
            "读来不累。拾光书屋第 12 期共读选中了它，店长的推荐语是：适合碎片"
            "时间随手翻几页，通勤路上读它最合适，到站刚好读完一节。"
        ),
    },
    {
        "book": "山月记",
        "category": "小说",
        "text": (
            "《山月记》是中岛敦的短篇小说，讲唐代诗人李征恃才傲物又极度自卑，"
            "不肯融入俗世，最终在疯狂与羞耻中化作猛虎的故事。书中「我深怕自己"
            "本非美玉，故而不敢加以刻苦琢磨」一句，是拾光书屋摘抄墙上被抄录最多"
            "的话。篇幅很短，一个晚上就能读完，适合下班后一口气看完。"
        ),
    },
    {
        "book": "小王子",
        "category": "童话",
        "text": (
            "《小王子》讲 B-612 小行星上的飞行员与玫瑰和狐狸的故事，核心词是"
            "「驯养」——狐狸说，你为你的玫瑰花费的时间，使你的玫瑰变得如此重要。"
            "拾光书屋每周日下午三点在儿童区办「亲子共读《小王子》」故事会，"
            "家长可以带孩子免费参加，无需报名。"
        ),
    },
    {
        "book": "城南旧事",
        "category": "小说",
        "text": (
            "《城南旧事》是林海音的自传体小说，借小女孩英子的眼睛看二十世纪"
            "二十年代的老北京：骆驼队、惠安馆、爸爸的花儿落了。文字干净克制，"
            "怀旧但不滥情。店长评价：适合找一个安静的午后，泡一壶茶慢慢读，"
            "读完你会想给老朋友打个电话。"
        ),
    },
    {
        "book": "店内公告",
        "category": "公告",
        "text": (
            "拾光书屋营业时间为每天 10:00 至 22:00，周一不闭店。会员一次最多"
            "可借 5 本书，借期 14 天，可续借一次；逾期每本书每天收取 0.5 元"
            "迟还金。店内咖啡区谢绝外带饮品，自习区晚上 18 点后请保持安静。"
        ),
    },
    {
        "book": "活动安排",
        "category": "公告",
        "text": (
            "拾光书屋读书会固定在每周五晚 19:00 举行，本月主题是「短篇之夜」，"
            "将共读《山月记》和《变形记》。每期限 12 人，请提前在前台或会员群"
            "里报名，免费参加。周日 15:00 是儿童故事会，直接带孩子来即可。"
        ),
    },
]

RAG_SYSTEM_TEMPLATE: str = (
    "你是拾光书屋的店员，只依据【书屋笔记】回答读者的问题；"
    "笔记里没有的信息就直说「书屋笔记里没有相关内容」，不要编造。"
    "回答用简体中文，简洁、客观。\n\n"
    "【书屋笔记】\n{context}"
)

REWRITE_TEMPLATE: str = (
    "你是拾光书屋的检索助手。为了在书屋笔记里找到更多相关内容，"
    "把读者的问题改写成 3 种不同的说法：换角度、换用词，但不要改变原意。"
    "每行一个，只输出改写后的问题，不要编号、不要解释。\n\n"
    "读者的问题：{question}"
)

# ----------------------------- 基础工具 -----------------------------
def ollama_alive(timeout_s: float = 2.0) -> bool:
    """探测本地 Ollama 服务是否在线。"""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout_s) as resp:
            return resp.status == 200
    except OSError:
        return False


class BagOfWordsEmbeddings(Embeddings):
    """离线兜底嵌入：把相邻两个字组成「词」丢进袋子（字符二元组）再归一化。

    语义能力为零，但足以把「字面重叠多」的笔记排到前面，
    让切割→入库→检索这条流水线在离线时也能完整演示。
    """

    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> int:
        """稳定哈希：同一 token 在任何进程里都落到同一个桶。"""
        return zlib.crc32(token.encode("utf-8")) % self.dim

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = [text[i : i + 2] for i in range(len(text) - 1)]  # 字符二元组
        for token in tokens:
            vec[self._bucket(token)] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def build_corpus_docs() -> list[Document]:
    """把硬编码语料转成 Document 列表，metadata 记好书名与类别。"""
    return [
        Document(page_content=note["text"], metadata={"book": note["book"], "category": note["category"]})
        for note in READING_NOTES
    ]


def format_docs(docs: list[Document]) -> str:
    """把文档列表拼成一段纯文本，塞进模板变量（RAG 链的固定零件）。"""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_prompt() -> ChatPromptTemplate:
    """RAG 问答提示词：system 放笔记，human 放问题。"""
    return ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_TEMPLATE),
        ("human", "{question}"),
    ])


# ----------------------------- 演示步骤 -----------------------------
def demo_retrieval(
    vectorstore: InMemoryVectorStore, question: str, score_threshold: float
) -> None:
    """课件第 2、3 节：k / 分数阈值 / MMR / filter。"""
    print("=" * 62)
    print("== 2) 检索参数：k / 分数阈值 / MMR / filter ==")
    print(f"问题：{question}")

    # 2.1 k：普通相似度检索，一次取 TOP_K 条
    retriever_k = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    print(f"\n-- 普通 similarity 检索（k={TOP_K}）--")
    for doc in retriever_k.invoke(question):
        print(f"  [{doc.metadata['book']}] {doc.page_content[:24]}…")

    # 2.2 分数阈值：InMemoryVectorStore 不支持 search_type="similarity_score_threshold"
    #     （会抛 NotImplementedError），标准替代做法是拿到分数后自己过滤
    print(f"\n-- 相似度分数 + 阈值 {score_threshold} 过滤（k 先放大到 6）--")
    hits: list[tuple[Document, float]] = vectorstore.similarity_search_with_score(question, k=6)
    kept = 0
    for doc, score in hits:
        mark = "保留" if score >= score_threshold else "丢弃"
        print(f"  分数 {score:.3f} [{mark}] [{doc.metadata['book']}] {doc.page_content[:18]}…")
        kept += score >= score_threshold
    if kept == 0:
        print("  （一条都没过线——阈值要按嵌入模型的分数分布重新校准）")

    # 2.3 MMR：先海选 fetch_k 条，再兼顾相关性与多样性精选 TOP_K 条
    retriever_mmr = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 10, "lambda_mult": 0.5},
    )
    print(f"\n-- MMR 检索（fetch_k=10 里精挑 {TOP_K} 条）--")
    for doc in retriever_mmr.invoke(question):
        print(f"  [{doc.metadata['book']}] {doc.page_content[:24]}…")

    # 2.4 filter：按 metadata 预筛。注意 InMemoryVectorStore 的 filter 是函数，不是 dict
    category_filter: Callable[[Document], bool] = (
        lambda doc: doc.metadata.get("category") == "小说"
    )
    retriever_filter = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K, "filter": category_filter}
    )
    print("\n-- filter：只在「小说」类笔记里检索 --")
    for doc in retriever_filter.invoke(question):
        print(f"  [{doc.metadata['book']}] {doc.page_content[:24]}…")
    print()


def demo_rag_chain(retriever, model: ChatOllama, question: str) -> None:
    """课件第 4、5 节：经典三段式 RAG 链。"""
    print("=" * 62)
    print("== 3) LCEL 组装 RAG 链：RunnableParallel → prompt → model → parser ==")
    chain = (
        RunnableParallel(
            context=retriever | format_docs,  # 一手取笔记
            question=RunnablePassthrough(),   # 一手递问题
        )
        | build_rag_prompt()
        | model
        | StrOutputParser()
    )
    print(f"问题：{question}")
    print("回答：", chain.invoke(question))
    print()


def demo_rag_chain_offline(retriever, question: str) -> None:
    """离线版：链组装到 prompt 为止，展示填充后的提示词。"""
    print("=" * 62)
    print("== 3) 组装 RAG 链（离线：只跑到「提示词」这一环）==")
    prompt_only = RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
    ) | build_rag_prompt()
    print("提示：下面是链条中间产物「填充完毕的提示词」，"
          "在线模式下再接 | model | StrOutputParser() 就是完整 RAG。")
    print("-" * 62)
    print(prompt_only.invoke(question).to_string())
    print("-" * 62)
    print()


def demo_rag_with_sources(retriever, model: ChatOllama, question: str) -> None:
    """课件第 6 节：来源溯源——答案连同被引用文档一起返回。"""
    print("=" * 62)
    print("== 4) 来源溯源：答案 + 引用列表 ==")
    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | build_rag_prompt()
        | model
        | StrOutputParser()
    )
    rag_with_sources = RunnableParallel(
        context=retriever,               # 保留原始 Document 列表
        question=RunnablePassthrough(),
    ).assign(answer=answer_chain)        # 拿着上面的 dict 并行算答案

    result = rag_with_sources.invoke(question)
    print(f"问题：{result['question']}")
    print(f"回答：{result['answer']}")
    print("引用来源：")
    for i, doc in enumerate(result["context"], start=1):
        print(f"  {i}. [{doc.metadata['book']}] {doc.page_content[:26]}…")
    print()


def demo_multi_query(retriever, model: ChatOllama, question: str) -> None:
    """课件第 7 节：多查询检索——改写问题、分头检索、合并去重。"""
    print("=" * 62)
    print("== 5) 多查询检索：一个问题，多种问法 ==")
    rewriter = PromptTemplate.from_template(REWRITE_TEMPLATE) | model | StrOutputParser()

    raw = rewriter.invoke({"question": question})
    variants = [line.strip() for line in raw.splitlines() if line.strip()]
    print("模型改写出的问法：")
    for v in variants:
        print(f"  - {v}")

    # 原问题 + 每条改写各检索 2 条，合并去重（去重键优先用文档 id）
    seen: set[str] = set()
    merged: list[Document] = []
    for q in [question, *variants]:
        for doc in retriever.invoke(q)[:2]:
            key = doc.id or doc.page_content
            if key not in seen:
                seen.add(key)
                merged.append(doc)

    print(f"合并去重后共 {len(merged)} 条：")
    for doc in merged:
        print(f"  [{doc.metadata['book']}] {doc.page_content[:24]}…")
    print()


# ----------------------------- 主流程 -----------------------------
def main() -> None:
    print("拾光书屋 · 第 08 章 demo：检索与 RAG")
    print("=" * 62)

    online = ollama_alive()
    model: ChatOllama | None
    if online:
        print(f"[在线模式] 检测到 Ollama（{OLLAMA_BASE_URL}），使用真实嵌入与对话模型。")
        embeddings: Embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        # qwen3 默认会在正文里输出 <think> 标签，reasoning=False 直接关掉思考
        model = ChatOllama(model=CHAT_MODEL, temperature=0, reasoning=False)
    else:
        print("[离线模式] 未检测到 Ollama，嵌入降级为内置词袋向量，")
        print("           仅演示检索部分；RAG 问答与多查询改写需要模型，已跳过。")
        print(f"           请启动 Ollama 并执行 `ollama pull {CHAT_MODEL}`、"
              f"`ollama pull {EMBED_MODEL}` 后重跑。")
        embeddings = BagOfWordsEmbeddings()
        model = None

    # 1) 语料 → 切割 → 嵌入 → 入库
    print("=" * 62)
    print("== 1) 语料 → 切割 → 嵌入 → 入库 ==")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", " ", ""],
    )
    corpus = build_corpus_docs()
    chunks = splitter.split_documents(corpus)
    print(f"共 {len(corpus)} 篇笔记 → 切成 {len(chunks)} 个小块（"
          f"chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）")

    vectorstore = InMemoryVectorStore(embedding=embeddings)
    vectorstore.add_documents(chunks)  # 嵌入在这一步发生
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    print()

    # 2) 检索参数演示（在线/离线都跑）
    threshold = SCORE_THRESHOLD_ONLINE if online else SCORE_THRESHOLD_OFFLINE
    demo_retrieval(vectorstore, "书屋的营业时间是几点？借书最多能借几本？", threshold)

    # 3~5) 需要模型的部分
    if model is not None:
        demo_rag_chain(retriever, model, "书屋的营业时间是几点？借书最多能借几本？")
        demo_rag_with_sources(retriever, model, "会员一次能借几本书？")
        demo_multi_query(retriever, model, "上班族在地铁上适合读什么书？")
    else:
        demo_rag_chain_offline(retriever, "书屋的营业时间是几点？借书最多能借几本？")


if __name__ == "__main__":
    main()
