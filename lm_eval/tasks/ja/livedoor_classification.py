"""
Livedoor News Classification

Livedoor ニュースの各記事について、その記事がどのカテゴリに属するかを分類するタスクです。

プロンプトとして、marc_ja をもとに設定しています。

Homepage: https://www.rondhuit.com/download.html#ldcc
"""
import datasets

from lm_eval.base import MultipleChoiceTask, rf

_CITATION = """
"""



class LivedoorClassificationWithFintanPrompt(MultipleChoiceTask):
    """
    prompt template is taken from [ChatGPT vs BERT: どちらが日本語をより理解できるのか?](https://fintan.jp/page/9126/)
    """
    VERSION = 1.0
    PROMPT_VERSION = 0.0
    DATASET_PATH = "retrieva-jp/livedoor-news-classification"
    DATASET_NAME = None
    DESCRIPTION = "ニュース記事をmove-enter、it-life-hack、kaden-channel、"\
        "topic-news、livedoor-homme、peachy、sports-watch、dokujo-tsushin、smax"\
        "のいずれかのカテゴリに分類してください。出力は小文字化してください。 \n\n"
    # livedoor classification はおそらく以下のid2category
    # {0: "movie-enter", 1: "it-life-hack", 2: "kaden-channel", 3: "topic-news", 4: "livedoor-homme", 5: "peachy", 6: "sports-watch", 7: "dokujo-tsushin", 8: "smax"}
    CHOICES = ["movie-enter", "it-life-hack", "kaden-channel", "topic-news", "livedoor-homme", "peachy", "sports-watch", "dokujo-tsushin", "smax"]

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        """Download for livedoor classification task."""
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_state=42,  # for reproducibility, fix it.
            shuffle=True,
        )

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
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        return {
            "query": doc["content"],
            "choices": self.CHOICES,
            "gold": int(doc["category"]),
        }

    def doc_to_text(self, doc):
        """
        製品レビュー:{query}
        センチメント:
        """
        return (
            f"ニュース記事:{doc['query']}\n"
            "カテゴリ:"
        )

    def doc_to_target(self, doc):
        return doc["choices"][doc["gold"]]

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, "{}".format(choice))[0] for choice in doc["choices"]
        ]

        return lls



class LivedoorClassificationWithJAAlpacaPrompt(LivedoorClassificationWithFintanPrompt):
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
    PROMPT_VERSION = 0.1
    DESCRIPTION = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
    INSTRUCTION = "以下のニュース記事を、movie-enter、ITライフハック、家電チャンネル、"\
        "トピックニュース、livedoor-homme、Peachy、sports-watch、独女通信、s-maxの"\
        "カテゴリクラスのいずれかに分類してください。"
    CHOICES = ["movie-enter", "ITライフハック", "家電チャンネル", "トピックニュース", "livedoor-homme", "Peachy", "sports-watch", "独女通信", "s-max"]

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



class LivedoorClassificationWithRinnaInstructionSFT(LivedoorClassificationWithFintanPrompt):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """
    PROMPT_VERSION = 0.2
    DESCRIPTION = "ユーザー: 与えられたニュース記事を、movie-enter、ITライフハック、"\
        "家電チャンネル、トピックニュース、livedoor-homme、Peachy、sports-watch、"\
        "独女通信、s-maxのカテゴリクラスのいずれかに分類してください。<NL>システム: 分かりました。<NL>"
    CHOICES = ["movie-enter", "ITライフハック", "家電チャンネル", "トピックニュース", "livedoor-homme", "Peachy", "sports-watch", "独女通信", "s-max"]
    SEP = "<NL>"
    FEWSHOT_SEP = "<NL>"

    def doc_to_text(self, doc):
        input_text = doc['query']
        return f"ユーザー: {input_text}{self.SEP}システム: "



VERSIONS = [
    LivedoorClassificationWithFintanPrompt,
    LivedoorClassificationWithJAAlpacaPrompt,
    LivedoorClassificationWithRinnaInstructionSFT,
]


def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[f"livedoor_classification-{version_class.VERSION}-{version_class.PROMPT_VERSION}"] = version_class
    return tasks
