"""
PC Customer Demo

分類タスクです。

プロンプトとして、marc_ja をもとに設定しています。

Homepage: None
"""
from lm_eval.base import MultipleChoiceTask, rf

_CITATION = """
"""



class PcCustomerDemoWithFintanPrompt(MultipleChoiceTask):
    """
    prompt template is taken from [ChatGPT vs BERT: どちらが日本語をより理解できるのか?](https://fintan.jp/page/9126/)
    """
    VERSION = 1.0
    PROMPT_VERSION = 0.2
    DATASET_PATH = "retrieva-jp/pc-customer-demo"
    DATASET_NAME = None
    DESCRIPTION = "お問合せ文書を故障、破損、使用方法、新製品のいずれかのカテゴリに分類してください。 \n\n"
    CHOICES = ["故障", "破損", "使用方法", "新製品"]

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        return {
            "query": doc["text"],
            "choices": self.CHOICES,
            "gold": doc["category"],
        }

    def doc_to_text(self, doc):
        """
        製品レビュー:{query}
        センチメント:
        """
        return (
            f"お問合せ文書:{doc['query']}\n"
            "カテゴリ:"
        )

    def doc_to_target(self, doc):
        return doc["gold"]

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, "{}".format(choice))[0] for choice in doc["choices"]
        ]

        return lls



class PcCustomerDemoWithJAAlpacaPrompt(PcCustomerDemoWithFintanPrompt):
    """
    This prompt format was inspired by the below data in fujiki/japanese_alpaca_data.
    ```
    {
        'instruction': '以下のテキストを、ポジティブまたはネガティブの感情クラスのいずれかに分類してください。',
        'input': '製品が遅すぎて使い勝手が悪かったので、あまり好きではありませんでした。',
        'output': 'ネガティブ。'
    }
    ```
    Reference:
    - data: https://huggingface.co/datasets/fujiki/japanese_alpaca_data
    - code: https://github.com/Stability-AI/gpt-neox/blob/c130a4edc1120dccec8f02a34eb60d3e8f484cd3/finetune/finetune_base_ja.py#LL118C23-L127C11
    """
    PROMPT_VERSION = 0.3
    DESCRIPTION = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
    INSTRUCTION = "以下のお問合せ文書を、故障、破損、使用方法、新製品のカテゴリクラスのいずれかに分類してください。"

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
        input_text = doc['query']
        return f"### 指示:\n{self.INSTRUCTION}\n\n### 入力:\n{input_text}\n\n### 応答:\n"



class PcCustomerDemoWithRinnaInstructionSFT(PcCustomerDemoWithFintanPrompt):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """
    PROMPT_VERSION = 0.4
    DESCRIPTION = "ユーザー: 与えられたお問合せ文書を、故障、破損、使用方法、新製品のカテゴリクラスのいずれかに分類してください。<NL>システム: 分かりました。<NL>"
    SEP = "<NL>"
    FEWSHOT_SEP = "<NL>"

    def doc_to_text(self, doc):
        input_text = doc['query']
        return f"ユーザー: {input_text}{self.SEP}システム: "



VERSIONS = [
    PcCustomerDemoWithFintanPrompt,
    PcCustomerDemoWithJAAlpacaPrompt,
    PcCustomerDemoWithRinnaInstructionSFT,
]


def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[f"pc_customer_demo-{version_class.VERSION}-{version_class.PROMPT_VERSION}"] = version_class
    return tasks
