"""
src/predict.py
面向用户的英文句子 NER 预测入口，支持 CRF / BERT 两种后端。
负责成员：A（接口与集成）；B/C 各自实现 predict_tags()。

接口约定（对应 plan.md 5.4）：
- predict_tags(tokens: List[str]) -> List[str]  # 每个输入词对应一个 BIO 标签
- predict(text: str) -> List[{"text": ..., "type": ...}]

使用示例：
    from predict import predict
    entities = predict("I study at City University of Hong Kong.", model_type="crf")
"""

import json
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))

from spans import (
    fix_bio_sequence,
    bio_to_spans,
    tokenize_with_positions,
    entities_from_text,
)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# 抽象基类：模型预测接口（B/C 各自实现）
# ---------------------------------------------------------------------------

class BasePredictor(ABC):
    """
    模型预测接口约定。

    B（CRF）和 C（BERT）各自实现 predict_tags()，
    A 通过统一入口调用并做 BIO 解码。
    """

    @abstractmethod
    def predict_tags(self, tokens: List[str]) -> List[str]:
        """
        对已分词的英文句子逐词标注 BIO 标签。

        Parameters
        ----------
        tokens : List[str]
            原始词序列，例如 ["City", "University", "of", "Hong", "Kong"]。

        Returns
        -------
        List[str]
            与 tokens 等长的 BIO 标签序列。
            无效 I-X 在 spans.fix_bio_sequence() 中统一修正，此处可不处理。
        """
        ...

    def predict_text(self, text: str) -> List[Dict[str, str]]:
        """
        对原始英文句子做端到端实体抽取。

        Parameters
        ----------
        text : str
            原始英文句子。

        Returns
        -------
        List[Dict[str, str]]
            [{"text": "City University of Hong Kong", "type": "ORG"}, ...]
        """
        tokens, positions = tokenize_with_positions(text)
        if not tokens:
            return []

        raw_tags = self.predict_tags(tokens)
        assert len(raw_tags) == len(tokens), (
            f"predict_tags() 返回长度 ({len(raw_tags)}) "
            f"与输入 tokens 长度 ({len(tokens)}) 不一致"
        )

        return entities_from_text(text, raw_tags, tokens, positions)


# ---------------------------------------------------------------------------
# CRF 预测器（B 负责训练与特征提取，A 提供加载框架）
# ---------------------------------------------------------------------------

class CRFPredictor(BasePredictor):
    """
    加载 B 训练好的 sklearn-crfsuite CRF 模型并做推理。

    特征函数加载优先级：
    1. 从 src/train_crf.py 导入 sentence_features()（B 的实现）
    2. 从 configs/crf_feature_fn.py 加载 sentence_features()
    3. 使用内置备用基础特征
    """

    def __init__(self, model_path: Path):
        try:
            import sklearn_crfsuite  # noqa: F401  检查依赖
        except ImportError:
            raise ImportError(
                "CRFPredictor 需要 sklearn-crfsuite，"
                "请先 pip install sklearn-crfsuite"
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"CRF 模型文件不存在: {self.model_path}")

        import joblib
        self.model = joblib.load(self.model_path)

        # 加载特征函数（B 提供，返回整句特征列表）
        self._feature_fn = self._load_feature_fn()

    def _load_feature_fn(self):
        """
        按优先级加载 sentence_features(tokens) -> list[dict]。
        """
        # 优先从 B 的 train_crf.py 导入
        try:
            from train_crf import sentence_features
            print("[CRFPredictor] 使用 train_crf.sentence_features()（B 的实现）")
            return sentence_features
        except ImportError:
            pass

        # 其次从 configs/crf_feature_fn.py 加载
        feature_file = PROJECT_ROOT / "configs" / "crf_feature_fn.py"
        if feature_file.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("crf_feature_fn", feature_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            if hasattr(mod, "sentence_features"):
                print(f"[CRFPredictor] 使用特征函数: {feature_file}")
                return mod.sentence_features

        print("[CRFPredictor] 未找到 B 的特征函数，使用内置基础特征。"
              " B 应提供 train_crf.sentence_features()。")
        return self._default_features

    @staticmethod
    def _default_features(tokens: List[str]) -> List[dict]:
        """内置备用特征：返回与 tokens 等长的特征字典列表。"""
        def word_feat(w: str) -> dict:
            low = w.lower()
            return {
                "lower": low,
                "is_upper": w.isupper(),
                "is_title": w.istitle(),
                "is_digit": w.isdigit(),
                "has_digit": any(c.isdigit() for c in w),
                "has_hyphen": "-" in w,
                "bias": 1.0,
                "prefix_2": low[:2],
                "prefix_3": low[:3],
                "suffix_2": low[-2:],
                "suffix_3": low[-3:],
            }

        basic = [word_feat(w) for w in tokens]
        features = []
        for i, w in enumerate(tokens):
            feat = dict(basic[i])
            if i == 0:
                feat["BOS"] = True
            else:
                feat.update({f"prev:{k}": v for k, v in basic[i - 1].items()})
            if i == len(tokens) - 1:
                feat["EOS"] = True
            else:
                feat.update({f"next:{k}": v for k, v in basic[i + 1].items()})
            features.append(feat)
        return features

    def predict_tags(self, tokens: List[str]) -> List[str]:
        """调用 B 的 CRF 模型进行序列标注。"""
        if not tokens:
            return []
        features = self._feature_fn(tokens)  # 返回 list[dict]
        # sklearn-crfsuite 的 predict_single() 接受单句特征列表
        return self.model.predict_single(features)


# ---------------------------------------------------------------------------
# BERT 预测器（C 负责训练，A 提供加载框架）
# ---------------------------------------------------------------------------

class BERTPredictor(BasePredictor):
    """
    加载 C 微调好的 BERT 词分类模型，子词对齐后返回词级 BIO 标签。

    子词对齐规则（对应 plan.md 5.3）：
    - 使用 tokenizer 的 word_ids() 将子词映射回原始词索引。
    - 只取每个词的第一个子词的预测标签作为词级标签。
    - 忽略特殊 token（word_id 为 None 的位置）。
    """

    def __init__(
        self,
        model_path: Path,
        tokenizer_path: str = "google-bert/bert-base-cased",
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForTokenClassification
        except ImportError:
            raise ImportError(
                "BERTPredictor 需要 PyTorch 和 Transformers，"
                "请先 pip install torch transformers"
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"BERT 模型目录不存在: {self.model_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_length = max_length
        self.torch = torch

        print(f"[BERTPredictor] 设备: {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_path
        ).to(device).eval()

        # 加载标签映射（C 训练时保存 config.json 中含 id2label）
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(self.model_path)
        self.id2label: Dict[int, str] = {
            int(k): v for k, v in cfg.id2label.items()
        }

    def predict_tags(self, tokens: List[str]) -> List[str]:
        """
        对词级 tokens 做 BERT 推理，返回 BIO 标签序列。

        子词对齐：word_ids() 返回子词对应的原词索引；
        特殊 token 的 word_id 为 None，跳过；
        每个词取第一个子词的标签。
        """
        from torch.utils.data import DataLoader, TensorDataset

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with self.torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # logits shape: (1, seq_len, n_labels)
        pred_ids = outputs.logits.argmax(dim=-1)[0].tolist()  # type: ignore

        word_ids = encoding.word_ids(batch_index=0)

        tags: List[str] = []
        seen_word: set = set()

        for token_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue          # 特殊 token，跳过
            if word_id not in seen_word:
                # 该词的第一个子词：取其预测标签
                label_id = pred_ids[token_idx]
                tags.append(self.id2label.get(label_id, "O"))
                seen_word.add(word_id)

        # 若截断导致部分词无预测，补充 'O'
        while len(tags) < len(tokens):
            tags.append("O")

        return tags[:len(tokens)]


# ---------------------------------------------------------------------------
# 顶层接口：predict()
# ---------------------------------------------------------------------------

def get_predictor(model_type: str, model_path: Optional[Path] = None, **kwargs) -> BasePredictor:
    """
    根据模型类型加载对应的预测器。

    Parameters
    ----------
    model_type : str
        'crf' 或 'bert'。
    model_path : Path, optional
        模型路径。未指定时使用 models/ 下的默认文件名。
    **kwargs
        传递给具体 Predictor 构造器的额外参数（如 device、tokenizer_path）。

    Returns
    -------
    BasePredictor
    """
    model_type = model_type.lower()

    if model_type == "crf":
        default_path = MODELS_DIR / "crf_full.joblib"
        path = model_path or default_path
        return CRFPredictor(path)

    elif model_type == "bert":
        default_path = MODELS_DIR / "bert_full"
        path = model_path or default_path
        return BERTPredictor(path, **kwargs)

    else:
        raise ValueError(f"未知模型类型: {model_type!r}，可选: 'crf' / 'bert'")


def predict(
    text: str,
    model_type: str = "crf",
    model_path: Optional[Path] = None,
    **kwargs,
) -> List[Dict[str, str]]:
    """
    顶层用户接口：输入原始英文句子，返回实体列表。

    Parameters
    ----------
    text : str
        原始英文句子。
    model_type : str
        'crf' 或 'bert'。
    model_path : Path, optional
        自定义模型路径。

    Returns
    -------
    List[Dict[str, str]]
        [{"text": "...", "type": "PER|ORG|LOC|MISC"}, ...]
    """
    predictor = get_predictor(model_type, model_path, **kwargs)
    return predictor.predict_text(text)


def predict_loaded(text: str, predictor: BasePredictor) -> List[Dict[str, str]]:
    """
    使用已加载的 predictor 做预测，避免重复加载模型。
    适合批量推理或演示场景。
    """
    return predictor.predict_text(text)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="英文 NER 预测入口（A 负责）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python predict.py "I study at City University of Hong Kong." --model crf
  python predict.py "Barack Obama was born in Hawaii." --model bert
  python predict.py "..." --model crf --path models/crf_10pct.joblib
        """,
    )
    parser.add_argument("text", help="待预测的英文句子")
    parser.add_argument("--model", choices=["crf", "bert"], default="crf",
                        help="模型类型（默认 crf）")
    parser.add_argument("--path", type=Path, default=None,
                        help="自定义模型路径")
    parser.add_argument("--device", default=None,
                        help="BERT 推理设备，如 'cpu' / 'cuda'")
    parser.add_argument("--json", action="store_true",
                        help="以 JSON 格式输出（默认人类可读）")

    args = parser.parse_args()

    kwargs = {}
    if args.device:
        kwargs["device"] = args.device

    entities = predict(args.text, model_type=args.model,
                       model_path=args.path, **kwargs)

    if args.json:
        print(json.dumps(entities, ensure_ascii=False, indent=2))
    else:
        print(f"\n输入: {args.text!r}")
        print(f"模型: {args.model}")
        if not entities:
            print("识别结果: （无实体）")
        else:
            print("识别结果:")
            for e in entities:
                print(f"  [{e['type']:4s}] {e['text']}")
