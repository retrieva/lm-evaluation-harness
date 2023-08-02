"""
Evidence-based Explanation Dataset (Identification)
https://anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P1-8.pdf
https://aclanthology.org/2020.aacl-main.89/

レビュータイトルに対してレビュー本文が evidence になっているか判定するタスクです。
例として、「景色が良い」というタイトルに対して、レビュー文が「部屋から海が見えます」の場合、
ラベルとしては evidence になります。

プロンプトとして、jnli をもとに設定しています。

Homepage: https://github.com/megagonlabs/ebe-dataset
"""
from lm_eval.base import MultipleChoiceTask, rf

_CITATION = """
@inproceedings{kanouchi-etal-2020-may,
    title = "You May Like This Hotel Because ...: Identifying Evidence for Explainable Recommendations",
    author = "Kanouchi, Shin  and
      Neishi, Masato  and
      Hayashibe, Yuta  and
      Ouchi, Hiroki  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing",
    month = dec,
    year = "2020",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.aacl-main.89",
    pages = "890--899",
}
"""



class EbeIdentificationWithFintanPrompt(MultipleChoiceTask):
    """
    prompt template is taken from [ChatGPT vs BERT: どちらが日本語をより理解できるのか?](https://fintan.jp/page/9126/)
    """
    VERSION = 1.0
    PROMPT_VERSION = 0.0
    DATASET_PATH = "retrieva-jp/ebe_identification"
    DATASET_NAME = None
    DESCRIPTION = "前提と仮説の関係をentailment、contradictionのどちらかを回答してください。\n\n" + \
        "制約:\n" + \
        "- 前提から仮説が、論理的知識や常識的知識を用いて導出可能である場合はentailmentと出力\n" + \
        "- 前提と仮説が両立しえない場合、関係ない場合はcontradictionと出力\n\n"
    CHOICES = ["contradiction", "entailment"]

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
        """detokenizeを行いつつ、データの格納を行う

        このタスクでは前提をレビュー文、仮説をレビュータイトルとして扱う。
        """
        return {
            "premise": "".join(doc["sent_wakati"].split()),
            "hypothesis": "".join(doc["title_wakati"].split()),
            "choices": self.CHOICES,
            "gold": int(doc["target"]),
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



class EbeIdentificationWithJAAlpacaPrompt(EbeIdentificationWithFintanPrompt):
    """
    Reference:
    - data: https://huggingface.co/datasets/fujiki/japanese_alpaca_data
    - code: https://github.com/Stability-AI/gpt-neox/blob/c130a4edc1120dccec8f02a34eb60d3e8f484cd3/finetune/finetune_base_ja.py#LL118C23-L127C11
    """
    PROMPT_VERSION = 0.1
    DESCRIPTION = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"
    INSTRUCTION = f"与えられた前提と仮説の関係を回答してください。\n\n出力は以下から選択してください：\n" + "\n".join(EbeIdentificationWithFintanPrompt.CHOICES)

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



class EbeIdentificationWithRinnaInstructionSFT(EbeIdentificationWithFintanPrompt):
    """
    Reference:
    - HF Hub: https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
    """
    PROMPT_VERSION = 0.2
    DESCRIPTION = "ユーザー: " + f"与えられた前提と仮説の関係を回答してください。出力は以下から選択してください：<NL>" + "<NL>".join(EbeIdentificationWithFintanPrompt.CHOICES) + "<NL>システム: 分かりました。<NL>"
    SEP = "<NL>"
    FEWSHOT_SEP = "<NL>"

    def doc_to_text(self, doc):
        input_text = f"前提：{doc['premise']}\n仮説：{doc['hypothesis']}"
        return f"ユーザー: {input_text}{self.SEP}システム: "



VERSIONS = [
    EbeIdentificationWithFintanPrompt,
    EbeIdentificationWithJAAlpacaPrompt,
    EbeIdentificationWithRinnaInstructionSFT,
]


def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[f"ebe_identification-{version_class.VERSION}-{version_class.PROMPT_VERSION}"] = version_class
    return tasks
