"""
WikiLingua: A New Benchmark Dataset for Cross-Lingual Abstractive Summarization
https://aclanthology.org/2020.findings-emnlp.360/

We introduce WikiLingua, a large-scale, multilingual dataset for the evaluation of cross-lingual abstractive summarization systems. We extract article and summary pairs in 18 languages from WikiHow, a high quality, collaborative resource of how-to guides on a diverse set of topics written by human authors. We create gold-standard article-summary alignments across languages by aligning the images that are used to describe each how-to step in an article. As a set of baselines for further studies, we evaluate the performance of existing cross-lingual abstractive summarization methods on our dataset. We further propose a method for direct cross-lingual summarization (i.e., without requiring translation at inference time) by leveraging synthetic data and Neural Machine Translation as a pre-training step. Our method significantly outperforms the baseline approaches, while being more cost efficient during inference.

Homepage: https://github.com/esdurmus/Wikilingua
"""
import numpy as np
import datasets
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
from lm_eval.utils import bleu, rouge



_CITATION = """
@inproceedings{ladhak-etal-2020-wikilingua, title = "{W}iki{L}ingua: A New Benchmark Dataset for Cross-Lingual Abstractive Summarization", author = "Ladhak, Faisal and Durmus, Esin and Cardie, Claire and McKeown, Kathleen", booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020", month = nov, year = "2020", address = "Online", publisher = "Association for Computational Linguistics", url = "https://aclanthology.org/2020.findings-emnlp.360", doi = "10.18653/v1/2020.findings-emnlp.360", pages = "4034--4048", abstract = "We introduce WikiLingua, a large-scale, multilingual dataset for the evaluation of cross-lingual abstractive summarization systems. We extract article and summary pairs in 18 languages from WikiHow, a high quality, collaborative resource of how-to guides on a diverse set of topics written by human authors. We create gold-standard article-summary alignments across languages by aligning the images that are used to describe each how-to step in an article. As a set of baselines for further studies, we evaluate the performance of existing cross-lingual abstractive summarization methods on our dataset. We further propose a method for direct cross-lingual summarization (i.e., without requiring translation at inference time) by leveraging synthetic data and Neural Machine Translation as a pre-training step. Our method significantly outperforms the baseline approaches, while being more cost efficient during inference.", }
"""


# TODO make a summarization task
class Wikilingua(Task):
    VERSION = 1
    # custom prompt
    PROMPT_VERSION = 0.0
    DATASET_PATH = "GEM/wiki_lingua"
    DATASET_NAME = "ja"
    DESCRIPTION = "与えられた文章を要約して下さい。\n\n"

    def __init__(self):
        super().__init__()

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]
    
    def train_docs(self):
        return self.dataset["train"]

    def doc_to_text(self, doc):
        return doc["source"]

    def doc_to_target(self, doc):
        target = doc["target"]

        #XXX: consider fixing weird formatting. In the targets it seems
        # inconsistent whether sentences are separated with "。 " or "\u3000 "
        # (\u3000 = full width space)

        #target = doc["target"].replace(" \u3000", "\u3000").replace("\u3000 ", "。")
        return target

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        completion = rf.greedy_until(ctx, ["\n"])
        return completion

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        completion = results[0].strip()

        ref = doc["source"]
        bleu_score = bleu([[ref]], [completion])
        rouge_scores = rouge([ref], [completion])


        return {
            "bleu": bleu_score,
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeLsum"],
        }

    def aggregation(self):
        return {
            "bleu": mean,
            "rouge1": mean,
            "rouge2": mean,
            "rougeL": mean,
        }

    def higher_is_better(self):
        return {
            "bleu": True,
            "rouge1": True,
            "rouge2": True,
            "rougeL": True,
        }
