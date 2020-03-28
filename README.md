# NLG Evaluation Metrics

A unified framework for recent evaluation metrics about natural language generation.

Work in progress.

## Installation

```bash
git clone git@github.com:yuhui-zh15/nlg_metrics.git
cd nlg_metrics
pip install -e .
```

## Get Started

```python
>>> from nlg_metrics import RougeScorer
>>> scorer = RougeScorer()
>>> scores = scorer.score(['This is a test sentence.'], ['This is another test sentence.'])
>>> print(scores)
```

## Progress

| Metric     | Progress | Paper                                                        |
| ---------- | -------- | ------------------------------------------------------------ |
| ROUGE      | COMPLETE | [ROUGE: A Package for Automatic Evaluation of Summaries](https://www.aclweb.org/anthology/W04-1013.pdf) |
| BERTScore  | COMPLETE | [BERTScore: Evaluating Text Generation With BERT](https://arxiv.org/pdf/1904.09675.pdf) |
| FactScore  | COMPLETE | [Evaluating the Factual Correctness for Abstractive Summarization](https://cs.stanford.edu/~yuhuiz/assets/reports/factual.pdf) |
| MoverScore | TODO     | [MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance](https://arxiv.org/abs/1909.02622) |
| BLEU       | TODO     | [BLEU: a Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040.pdf) |
| METEOR     | TODO     | [METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments](https://www.cs.cmu.edu/~alavie/METEOR/pdf/Banerjee-Lavie-2005-METEOR.pdf) |

