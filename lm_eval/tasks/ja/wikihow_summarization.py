"""
Wikihow Summarization
https://www.anlp.jp/proceedings/annual_meeting/2021/pdf_dir/P1-12.pdf

プロンプトとして xlsum_ja を参考にしています。

Homepage: https://github.com/Katsumata420/wikihow_japanese
"""
import os
import inspect
from lm_eval.utils import rouge2_mecab
from lm_eval.base import rf, Task


_CITATION = """
"""


DYNAMIC_MAX_LENGTH = os.getenv("DYNAMIC_MAX_LENGTH", "true").lower()


class WikihowSummarization(Task):
    """
    - Use ROUGE-2 as [PaLM 2](https://ai.google/static/documents/palm2techreport.pdf)
    - Use Mecab tokenizer for Japanese eval
    """
    VERSION = 1.0
    PROMPT_VERSION = 0.0
    DATASET_PATH = "retrieva-jp/wikihow_summarization"
    DATASET_NAME = None
    DESCRIPTION = "与えられたハウトゥ記事を要約してください。\n\n"
    LOAD_TOKENIZER = True
    SEP="\n"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from . import MecabTokenizer
        self.tokenizer = MecabTokenizer()

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return f"ニュース記事:{doc['src']}\n要約:"

    def doc_to_target(self, doc):
        return doc["tgt"]

    def preprocess_ctx(self, ctx, max_length, ctx_prompt="ハウトゥ記事:", summary_prompt="要約:"):
        if len(self._tokenize(ctx)) <= max_length:
            return ctx
        # if the inputs too long, truncate inputs
        ctxs = [f"{ctx_prompt}{c}"for c in ctx.split(ctx_prompt)]
        description = ""
        if summary_prompt not in ctxs[0]:
            description = ctxs[0].replace(ctx_prompt, "")
            ctxs = ctxs[1:]
        max_length_per_shot = max_length // len(ctxs)
        res = description
        for c in ctxs:
            text, summary = c.split(summary_prompt)
            sentences = text.split("。")
            c_res = ""
            add_sentences = []
            for s in sentences:
                tmp = add_sentences + [s]
                if len(self._tokenize(text="。".join(tmp))) > max_length_per_shot:
                    if len(add_sentences) > 0:
                        add_sentences[-1] += "。"+self.SEP
                    else:
                        # I believe this case does't happen. But, let's make sure to avoid IndexError
                        # In this case, just truncate the first sentence
                        token_ids = self._tokenize(s)[:max_length_per_shot]
                        truncated_s = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                        add_sentences.append(truncated_s+self.SEP)
                    break
                add_sentences.append(s)
            c_res += "。".join(add_sentences)
            res += f"{c_res}{summary_prompt}{summary}"
        return res

    def _tokenize(self, text, **kwargs):
        encode_fn = self.tokenizer.encode
        if "add_special_tokens" in inspect.getfullargspec(encode_fn).args:
            encode_params = dict(add_special_tokens=False)
        else:
            encode_params = {}
        return encode_fn(text, **encode_params, **kwargs)

    def construct_requests(self, doc, ctx):
        if DYNAMIC_MAX_LENGTH == "false" or not hasattr(self.tokenizer, "encode"):
            max_num_tokens = self.max_gen_toks
        else:
            # length + some buffers (10)
            max_num_tokens = len(self._tokenize(doc["tgt"])) + 10
        ctx = self.preprocess_ctx(ctx, max_length=self.max_length-max_num_tokens)
        continuation = rf.greedy_until(ctx, [self.SEP], max_num_tokens)
        return continuation

    def process_results(self, doc, results):
        continuation = results[0]
        ground_truth = doc["tgt"]
        return {
            "rouge2": (
                continuation,
                ground_truth,
            )
        }

    def aggregation(self):
        return {
            "rouge2": self._rouge
        }

    def higher_is_better(self):
        return {
            "rouge2": True,
        }

    def _rouge(self, item):
        predictions, references = zip(*item)
        res = rouge2_mecab(refs=references, preds=predictions, tokenizer=self.tokenizer)
        return res["rouge2"]


class WikihowSummarizationWithJAAlpacaPrompt(WikihowSummarization):
    PROMPT_VERSION = 0.3
    DESCRIPTION = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
    INSTRUCTION = "与えられたハウトゥ記事を要約してください。"
    def doc_to_text(self, doc):
        """
        以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

        ### 指示:
        {instruction}

        ### 入力:
        {input}

        ### 応答:
        {response}
        """
        input_text = f"ハウトゥ記事:{doc['src']}"
        return f"### 指示:\n{self.INSTRUCTION}\n\n### 入力:\n{input_text}\n\n### 応答:\n"

    def preprocess_ctx(self, ctx, max_length):
        return super().preprocess_ctx(ctx, max_length, ctx_prompt=f"### 指示:\n{self.INSTRUCTION}\n\n### 入力:\n", summary_prompt="### 応答:\n")


class WikihowSummarizationWithRinnaInstructionSFT(WikihowSummarization):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """
    PROMPT_VERSION = 0.4
    DESCRIPTION = "ユーザー: 与えられたハウトゥ記事を要約してください。<NL>システム: 分かりました。<NL>"
    SEP = "<NL>"
    FEWSHOT_SEP = "<NL>"

    def doc_to_text(self, doc):
        input_text = f"ハウトゥ記事:{doc['src']}"
        return f"ユーザー: {input_text}{self.SEP}システム: "

    def preprocess_ctx(self, ctx, max_length):
        ctx = super().preprocess_ctx(ctx, max_length, ctx_prompt=f"ユーザー: ", summary_prompt=f"{self.SEP}システム: ")
        ctx = ctx.replace("<NL><NL>", "<NL>")
        return ctx


VERSIONS = [
    WikihowSummarization,
    WikihowSummarizationWithJAAlpacaPrompt,
    WikihowSummarizationWithRinnaInstructionSFT,
]


def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[f"wikihow_summarization-{version_class.VERSION}-{version_class.PROMPT_VERSION}"] = version_class
    return tasks
