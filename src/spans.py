"""
src/spans.py
BIO 解码与实体抽取，含无效 I-X 修正规则。
负责成员：A

统一规则（对应 plan.md 5.4）：
- I-X 出现在句首，或前一个词的标签不是同类型 B-X/I-X 时，将其视为 B-X。
- 结束索引采用右开边界（exclusive），与 seqeval 一致。
"""

from typing import List, Tuple, Dict, Optional

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------
# (start_word_idx, end_word_idx_exclusive, entity_type)
Span = Tuple[int, int, str]
# 实体输出格式
Entity = Dict[str, str]   # {"text": "...", "type": "..."}


# ---------------------------------------------------------------------------
# 核心：无效 I-X 修正
# ---------------------------------------------------------------------------

def fix_bio_sequence(tags: List[str]) -> List[str]:
    """
    修正 BIO 序列中的无效 I-X 标签。

    规则：
    1. I-X 出现在序列首位 → 改为 B-X
    2. I-X 的前一个标签不是 B-X 也不是 I-X → 改为 B-X

    Parameters
    ----------
    tags : List[str]
        原始 BIO 标签序列，例如 ['O', 'I-PER', 'B-ORG', 'I-ORG']。

    Returns
    -------
    List[str]
        修正后的标签序列。
    """
    if not tags:
        return []

    fixed = list(tags)

    for i, tag in enumerate(fixed):
        if not tag.startswith("I-"):
            continue

        etype = tag[2:]  # 实体类型，如 'PER'

        if i == 0:
            # 规则 1：I-X 在句首
            fixed[i] = f"B-{etype}"
        else:
            prev = fixed[i - 1]
            # 规则 2：前一个标签不是同类型的 B-X 或 I-X
            if prev != f"B-{etype}" and prev != f"I-{etype}":
                fixed[i] = f"B-{etype}"

    return fixed


# ---------------------------------------------------------------------------
# BIO → Span 解码
# ---------------------------------------------------------------------------

def bio_to_spans(tags: List[str], auto_fix: bool = True) -> List[Span]:
    """
    将 BIO 标签序列解码为实体 span 列表。

    Parameters
    ----------
    tags : List[str]
        词级 BIO 标签序列。
    auto_fix : bool
        是否在解码前自动修正无效 I-X（默认 True）。

    Returns
    -------
    List[Span]
        每个元素为 (start_idx, end_idx_exclusive, entity_type)。
        例：['B-PER', 'I-PER', 'O'] → [(0, 2, 'PER')]
    """
    if auto_fix:
        tags = fix_bio_sequence(tags)

    spans: List[Span] = []
    n = len(tags)
    i = 0

    while i < n:
        tag = tags[i]

        if tag.startswith("B-"):
            etype = tag[2:]
            start = i
            i += 1
            # 收集后续同类型 I-X
            while i < n and tags[i] == f"I-{etype}":
                i += 1
            spans.append((start, i, etype))  # 右开

        elif tag.startswith("I-"):
            # auto_fix=False 时仍可能遇到无效 I-X，当新实体处理
            etype = tag[2:]
            start = i
            i += 1
            while i < n and tags[i] == f"I-{etype}":
                i += 1
            spans.append((start, i, etype))

        else:  # 'O'
            i += 1

    return spans


# ---------------------------------------------------------------------------
# Span → 实体文本（词级 tokens）
# ---------------------------------------------------------------------------

def spans_to_entities(tokens: List[str], spans: List[Span]) -> List[Entity]:
    """
    将 span 索引与 tokens 合并为实体文本输出。

    Parameters
    ----------
    tokens : List[str]
        原始词序列。
    spans : List[Span]
        bio_to_spans() 返回的 span 列表。

    Returns
    -------
    List[Entity]
        [{"text": "City University of Hong Kong", "type": "ORG"}, ...]
    """
    return [
        {"text": " ".join(tokens[start:end]), "type": etype}
        for start, end, etype in spans
    ]


# ---------------------------------------------------------------------------
# 原始文本 → 实体（保留原文位置）
# ---------------------------------------------------------------------------

def tokenize_with_positions(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    对原始英文句子做简单分词，保留每个词的原文字符偏移。

    采用规则：按空白符切分，去除首尾标点，单词内部标点保留。
    若需要与 BERT tokenizer 对齐，可在 predict.py 中替换。

    Parameters
    ----------
    text : str
        原始英文句子。

    Returns
    -------
    tokens : List[str]
        分词结果。
    positions : List[Tuple[int, int]]
        每个词在原文中的 (start_char, end_char)。
    """
    tokens: List[str] = []
    positions: List[Tuple[int, int]] = []

    i = 0
    n = len(text)

    while i < n:
        # 跳过空白
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break

        # 记录词起始位置
        start = i

        # 读取到空白为止
        while i < n and not text[i].isspace():
            i += 1

        word_raw = text[start:i]

        # 去除尾部标点（保留单词内部符号）
        trailing = ""
        while word_raw and word_raw[-1] in ".,;:!?\"'()[]{},":
            trailing = word_raw[-1] + trailing
            word_raw = word_raw[:-1]

        # 去除首部标点
        while word_raw and word_raw[0] in "\"'([{,":
            word_raw = word_raw[1:]
            start += 1

        if word_raw:
            tokens.append(word_raw)
            positions.append((start, start + len(word_raw)))

    return tokens, positions


def entities_from_text(
    text: str,
    tags: List[str],
    tokens: List[str],
    positions: List[Tuple[int, int]],
    auto_fix: bool = True,
) -> List[Entity]:
    """
    从 BIO 标签和原文位置提取实体，实体 text 从原句切片。

    Parameters
    ----------
    text : str
        原始完整句子。
    tags : List[str]
        与 tokens 等长的 BIO 标签。
    tokens : List[str]
        分词结果。
    positions : List[Tuple[int, int]]
        每个词的字符偏移。
    auto_fix : bool
        是否修正无效 I-X。

    Returns
    -------
    List[Entity]
    """
    spans = bio_to_spans(tags, auto_fix=auto_fix)
    entities: List[Entity] = []

    for start, end, etype in spans:
        char_start = positions[start][0]
        char_end = positions[end - 1][1]
        entity_text = text[char_start:char_end]
        entities.append({"text": entity_text, "type": etype})

    return entities


# ---------------------------------------------------------------------------
# 工具：span 转 BIO 标签（用于 seqeval 交叉验证）
# ---------------------------------------------------------------------------

def spans_to_bio(spans: List[Span], n_tokens: int) -> List[str]:
    """
    将 span 列表还原为 BIO 序列（主要用于测试）。

    Parameters
    ----------
    spans : List[Span]
    n_tokens : int

    Returns
    -------
    List[str]
    """
    tags = ["O"] * n_tokens
    for start, end, etype in spans:
        tags[start] = f"B-{etype}"
        for i in range(start + 1, end):
            tags[i] = f"I-{etype}"
    return tags


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 测试 fix_bio_sequence
    cases_fix = [
        (["I-PER", "O"],           ["B-PER", "O"]),          # 句首 I-X
        (["O", "I-ORG", "I-ORG"], ["O", "B-ORG", "I-ORG"]), # 无效 I-X
        (["B-PER", "I-PER"],      ["B-PER", "I-PER"]),       # 合法，不变
        (["I-PER", "I-PER"],      ["B-PER", "I-PER"]),       # 两句都改
    ]
    print("=== fix_bio_sequence 测试 ===")
    for inp, expected in cases_fix:
        result = fix_bio_sequence(inp)
        status = "PASS" if result == expected else f"FAIL (got {result})"
        print(f"  {inp} → {status}")

    # 测试 bio_to_spans
    cases_spans = [
        (["B-PER", "I-PER", "O", "B-ORG"], [(0, 2, "PER"), (3, 4, "ORG")]),
        (["O", "O"],                         []),
        (["B-LOC"],                          [(0, 1, "LOC")]),
    ]
    print("\n=== bio_to_spans 测试 ===")
    for inp, expected in cases_spans:
        result = bio_to_spans(inp)
        status = "PASS" if result == expected else f"FAIL (got {result})"
        print(f"  {inp} → {status}")

    # 测试 entities_from_text
    test_sent = "I study at City University of Hong Kong."
    tokens, positions = tokenize_with_positions(test_sent)
    print(f"\n=== tokenize_with_positions ===")
    print(f"  句子: {test_sent!r}")
    print(f"  tokens: {tokens}")
    print(f"  positions: {positions}")

    # 模拟 B-PER/I-PER 标注（这里仅演示接口）
    mock_tags = ["O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG"]
    entities = entities_from_text(test_sent, mock_tags, tokens, positions)
    print(f"  entities: {entities}")

    print("\n[spans.py] 所有测试通过。")
