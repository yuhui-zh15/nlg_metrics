"""
Score a list of hypotheses against a list of references.
"""
import argparse

from rouge import rouge_scorer
from rouge import scoring

class RougeScorer():
    """ A scorer wrapper with initialization. """
    def __init__(self, metrics=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_cf=False, stem=False, n_samples=1000, verbose=True):
        self.metrics = metrics
        self.use_cf = use_cf
        self.stem = stem
        self.n_samples = n_samples
        self.verbose = verbose
    
    @property
    def scorer(self):
        if not hasattr(self, '_scorer'):
            self._scorer = rouge_scorer.RougeScorer(self.metrics, use_stemmer=self.stem)
        return self._scorer

    @property
    def aggregator(self):
        if not hasattr(self, '_aggregator'):
            self._aggregator = scoring.BootstrapAggregator(n_samples=self.n_samples)
        return self._aggregator

    def score_each(self, hyp, ref):
        """ Score a single pair. Useful for RL finetuning. Always return the mid value. """
        assert isinstance(hyp, str)
        assert isinstance(ref, str)
        results =  self.scorer.score(ref, hyp)
        out = []
        for m in self.metrics:
            r = results[m]
            out += [r.fmeasure*100]
        if len(out) == 1:
            return out[0]
        return out

    def score(self, hypotheses, references):
        """
        Args:
            - hypotheses: a list of pretokenized hypotheses
            - references: a list of pretokenized references
        """
        assert len(hypotheses) == len(references)
        assert len(hypotheses) > 0
    
        if self.verbose:
            print("Calculating ROUGE scores for {} entries...".format(len(hypotheses)))
    
        for hyp, ref in zip(hypotheses, references):
            s = self.scorer.score(ref, hyp)
            self.aggregator.add_scores(s)
    
        results = self.aggregator.aggregate()
        self.aggregator.clear() # clear the storage buffer

        # return results
        out = []
        for m in self.metrics:
            r = results[m]
            if self.use_cf:
                out += [r.mid.fmeasure*100, r.low.fmeasure*100, r.high.fmeasure*100]
            else:
                out += [r.mid.fmeasure*100]
        if len(out) == 1:
            return out[0]
        return out

def load_txt(filename):
    data = []
    with open(filename) as infile:
        for line in infile:
            if len(line.rstrip()) == 0:
                continue
            data += [line]
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file", help="Filename with reference summaries.")
    parser.add_argument("pred_file", help="Filename with evaluated predictions.")
    args = parser.parse_args()

    gold = [s.split() for s in load_txt(args.gold_file)]
    pred = [s.split() for s in load_txt(args.pred_file)]
    assert len(gold) == len(pred), "Length of references and predictions must equal."

    print(f"Total number of summaries found: {len(gold)}")
    scorer = RougeScorer(use_cf=True)
    r1, r1_low, r1_hi, r2, r2_low, r2_hi, rl, rl_low, rl_hi = scorer.score(pred, gold)
    print("ROUGE results:\n")
    print("Metric\tScore\t95% CI-\t95% CI+")
    print("ROUGE-1\t{:.2f}\t{:.2f}\t{:.2f}".format(r1, r1_low-r1, r1_hi-r1))
    print("ROUGE-2\t{:.2f}\t{:.2f}\t{:.2f}".format(r2, r2_low-r2, r2_hi-r2))
    print("ROUGE-L\t{:.2f}\t{:.2f}\t{:.2f}".format(rl, rl_low-rl, rl_hi-rl))
   
if __name__ == '__main__':
    main()

