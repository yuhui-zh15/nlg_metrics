"""
Basic testing of part of speech tagging
"""

import pytest
from nlg_metrics import RougeScorer


def aeq(src, tgt, epsilon=1e-3):
    assert(type(src) == type(tgt)), f'type {type(src)} != type {type(tgt)}'
    if type(src) is list: assert(all(-epsilon < i - j < epsilon for i, j in zip(src, tgt))), f'{src} != {tgt}'
    else: assert(-epsilon < src - tgt < epsilon), f'{src} != {tgt}'


def test_rouge():
    # Test Case #1 for ROUGE
    scorer = RougeScorer()
    scores = scorer.score(['hello world'], ['hello world'])
    aeq(scores, [100])
