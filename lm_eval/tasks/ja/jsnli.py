"""
JSNLI: Japanese Stanford Natural Language Inference
https://ipsj.ixsq.nii.ac.jp/ej/index.php?active_action=repository_view_main_item_detail&page_id=13&block_id=8&item_id=206114&item_no=1

本データセットは自然言語推論 (NLI) の標準的ベンチマークである SNLIを日本語に翻訳したものです。SNLI に機械翻訳を適用した後、評価データにクラウドソーシングによる正確なフィルタリング、学習データに計算機による自動フィルタリングを施すことで構築されています。

プロンプトとして、jnli をもとに設定しています。

Homepage: https://nlp.ist.i.kyoto-u.ac.jp/?%E6%97%A5%E6%9C%AC%E8%AA%9ESNLI%28JSNLI%29%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88
"""
from lm_eval.base import MultipleChoiceTask, rf

_CITATION = """
"""



class JSNLIWithFintanPrompt(MultipleChoiceTask):
    """
    prompt template is taken from [ChatGPT vs BERT: どちらが日本語をより理解できるのか?](https://fintan.jp/page/9126/)
    """
    VERSION = 1.1
    PROMPT_VERSION = 0.2
    DATASET_PATH = "retrieva-jp/jsnli-v1.1"
    DATASET_NAME = None
    DESCRIPTION = "前提と仮説の関係をentailment、contradiction、neutralの中から回答してください。\n\n" + \
        "制約:\n" + \
        "- 前提から仮説が、論理的知識や常識的知識を用いて導出可能である場合はentailmentと出力\n" + \
        "- 前提と仮説が両立しえない場合はcontradictionと出力\n" + \
        "- そのいずれでもない場合はneutralと出力\n\n"
    CHOICES = ["entailment", "contradiction", "neutral"]

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["dev_gen"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test_gen"])

    def _process_doc(self, doc):
        return {
            "premise": doc["sentence_a"],
            "hypothesis": doc["sentence_b"],
            "choices": self.CHOICES,
            "gold": int(doc["label"]),
        }

    def doc_to_text(self, doc):
        """
        前提:{premise}
        仮説:{hypothesis}
        関係:
        """
        return (
            f"前提:{doc['premise']}\n"
            f"仮説:{doc['hypothesis']}\n"
            "関係:"
        )

    def doc_to_target(self, doc):
        return doc["choices"][doc["gold"]]

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, "{}".format(choice))[0] for choice in doc["choices"]
        ]
        return lls



class JSNLIWithJAAlpacaPrompt(JSNLIWithFintanPrompt):
    """
    Reference:
    - data: https://huggingface.co/datasets/fujiki/japanese_alpaca_data
    - code: https://github.com/Stability-AI/gpt-neox/blob/c130a4edc1120dccec8f02a34eb60d3e8f484cd3/finetune/finetune_base_ja.py#LL118C23-L127C11
    """
    PROMPT_VERSION = 0.3
    DESCRIPTION = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
    INSTRUCTION = f"与えられた前提と仮説の関係を回答してください。\n\n出力は以下から選択してください：\n" + "\n".join(JSNLIWithFintanPrompt.CHOICES)

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
        input_text = f"前提：{doc['premise']}\n仮説：{doc['hypothesis']}"
        return f"### 指示:\n{self.INSTRUCTION}\n\n### 入力:\n{input_text}\n\n### 応答:\n"



class JSNLIWithRinnaInstructionSFT(JSNLIWithFintanPrompt):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """
    PROMPT_VERSION = 0.4
    DESCRIPTION = "ユーザー: " + f"与えられた前提と仮説の関係を回答してください。出力は以下から選択してください：<NL>" + "<NL>".join(JSNLIWithFintanPrompt.CHOICES) + "<NL>システム: 分かりました。<NL>"
    SEP = "<NL>"
    FEWSHOT_SEP = "<NL>"

    def doc_to_text(self, doc):
        input_text = f"前提：{doc['premise']}\n仮説：{doc['hypothesis']}"
        return f"ユーザー: {input_text}{self.SEP}システム: "



VERSIONS = [
    JSNLIWithFintanPrompt,
    JSNLIWithJAAlpacaPrompt,
    JSNLIWithRinnaInstructionSFT,
]


def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[f"jsnli-{version_class.VERSION}-{version_class.PROMPT_VERSION}"] = version_class
    return tasks
