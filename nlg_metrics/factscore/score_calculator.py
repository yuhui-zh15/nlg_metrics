from typing import List, Dict
import numpy as np

class ScoreCalculator(object):
    def __init__(self, name):
        self.name = name
    
    def calculate(self, src_emb: np.array, tgt_emb: np.array) -> float:
        raise NotImplementedError

    def calculate_list(self, src_embs: List[np.array], tgt_embs: List[np.array]) -> List[float]:
        raise NotImplementedError


class DotProductScoreCalculator(ScoreCalculator):
    def __init__(self):
        super(DotProductScoreCalculator, self).__init__(name='DotProductScoreCalculator')
    
    def calculate(self, src_emb: np.array, tgt_emb: np.array) -> Dict:
        if src_emb is None or tgt_emb is None: return {'p': 0., 'r': 0., 'f1': 0., 'mat': None}
        src_emb, tgt_emb = self.normalize_emb(src_emb), self.normalize_emb(tgt_emb)
        mat = np.dot(src_emb, tgt_emb.T)
        p = np.mean(np.max(mat, axis=1))
        r = np.mean(np.max(mat, axis=0))
        f1 = 2 * p * r / (p + r)
        return {'p': p, 'r': r, 'f1': f1, 'mat': mat}

    def calculate_list(self, src_embs: List[np.array], tgt_embs: List[np.array]) -> List[Dict]:
        assert(len(src_embs) == len(tgt_embs))
        return [self.calculate(src_emb, tgt_emb) for (src_emb, tgt_emb) in zip(src_embs, tgt_embs)]

    def normalize_emb(self, emb):
        return emb / np.linalg.norm(emb, axis=1).reshape(-1, 1)

class DotProductWithThresholdScoreCalculator(ScoreCalculator):
    def __init__(self, threshold=0):
        super(DotProductWithThresholdScoreCalculator, self).__init__(name='DotProductWithThresholdScoreCalculator')
        self.threshold = threshold
    
    def calculate(self, src_emb: np.array, tgt_emb: np.array) -> Dict:
        if src_emb is None or tgt_emb is None: return {'p': 0., 'r': 0., 'f1': 0., 'mat': None}
        src_emb, tgt_emb = self.normalize_emb(src_emb), self.normalize_emb(tgt_emb)
        mat = np.dot(src_emb, tgt_emb.T)
        p = np.max(mat, axis=1)
        p_mask = (p > self.threshold)
        p = p * p_mask
        p = np.mean(p)
        r = np.max(mat, axis=0)
        r_mask = (r > self.threshold)
        r = r * r_mask
        r = np.mean(r)
        if p == 0 and r == 0: f1 = 0
        else: f1 = 2 * p * r / (p + r)
        return {'p': p, 'r': r, 'f1': f1, 'mat': mat}

    def calculate_list(self, src_embs: List[np.array], tgt_embs: List[np.array]) -> List[Dict]:
        assert(len(src_embs) == len(tgt_embs))
        return [self.calculate(src_emb, tgt_emb) for (src_emb, tgt_emb) in zip(src_embs, tgt_embs)]

    def normalize_emb(self, emb):
        return emb / np.linalg.norm(emb, axis=1).reshape(-1, 1)


