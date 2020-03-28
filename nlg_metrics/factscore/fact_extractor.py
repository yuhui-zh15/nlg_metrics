from typing import List
import functools
import operator
from allennlp.pretrained import open_information_extraction_stanovsky_2018
from allennlp.predictors.open_information_extraction import consolidate_predictions, join_mwp, make_oie_string, get_predicate_text
from collections import defaultdict
from pyopenie import OpenIE5

class FactExtractor(object):
    def __init__(self, name):
        self.name = name
    
    def extract(self, sent: str) -> List[List[str]]:
        raise NotImplementedError

    def extract_list(self, sents: List[str]) -> List[List[List[str]]]:
        raise NotImplementedError

class KnowItAllFactExtractor(FactExtractor):
    def __init__(self, cuda=-1):
        super(KnowItAllFactExtractor, self).__init__(name='KnowItAllFactExtractor')

    @property
    def extractor(self):
        if not hasattr(self, '_extractor'):
            print('Start loading KnowItAll...')
            # first: java -Xmx10g -XX:+UseConcMarkSweepGC -jar openie-assembly.jar --httpPort 8000 --split
            self._extractor = OpenIE5('http://localhost:8000')
            print('Finish loading KnowItAll...')
        return self._extractor
    
    def extract(self, sent: str) -> List[List[str]]:
        extractions = self.extractor.extract(sent)
        facts = []
        for extraction in extractions:
            extraction = extraction['extraction']
            arg1 = extraction['arg1']
            rel = extraction['rel']
            arg2s = extraction['arg2s']
            for arg2 in arg2s:
                facts.append((arg1['text'], rel['text'], arg2['text']))
        facts = [list(i) for i in set(facts)]
        return facts
    
    def extract_list(self, sents: List[str]) -> List[List[List[str]]]:
        facts_all = []
        for sent in sents:
            facts_all.append(self.extract(sent))
        return facts_all


class AllenNLPFactExtractor(FactExtractor):

    def __init__(self, cuda=-1):
        super(AllenNLPFactExtractor, self).__init__(name='AllenNLPFactExtractor')
        self.cuda = cuda

    @property
    def model(self):
        if not hasattr(self, '_model'):
            print('Start loading AllenNLP...')
            self._model = open_information_extraction_stanovsky_2018()
            if self.cuda >= 0: self._model._model.cuda(self.cuda)
            print('Finish loading AllenNLP...')
        return self._model

    def extract(self, sent: str) -> List[List[str]]:
        sents = self.split_sentence(sent)
        rets = []
        for s in sents:
            oie_inputs = self.create_instances(s)
            if len(oie_inputs) == 0: continue
            sent_preds = self.model.predict_batch_json(oie_inputs)
            triples = self.process_preds(sent_preds)
            rets += triples
        return rets

    def extract_list(self, sents: List[str]) -> List[List[List[str]]]:
        return [self.extract(sent) for sent in sents]

    def get_triples(self, extractions):
        triples = []
        for extraction in extractions:
            args = extraction.split('\t')
            s, r, t = None, None, None
            for arg in args:
                if arg.startswith('ARG0:'): s = arg.split(':', 1)[1]
                if arg.startswith('V:'): r = arg.split(':', 1)[1]
                if arg.startswith('ARG1:'): t = arg.split(':', 1)[1]
            triples.append([s, r, t])
        return triples

    def process_preds(self, preds):
        tags_all = []
        conf_all = []
        sent_tokens = None
        for outputs in preds:
            sent_tokens = outputs["words"]
            tags = outputs["tags"]
            class_probs = outputs["class_probabilities"]
            conf = self.get_confidence(tags, class_probs)
            tags_all.append(tags)
            conf_all.append(conf)
        extractions, _ = self.format_extractions([Mock_token(tok) for tok in sent_tokens], tags_all)
        triples = self.get_triples(extractions)
        return triples

    def split_sentence(self, sent: str, eos=['.', '!', '?']) -> List[str]:
        sent_tokens = [str(i) for i in self.model._tokenizer.tokenize(sent)] # <TODO>: 2x tokenization, better ways required
        sents = []
        last_i = 0
        for i, token in enumerate(sent_tokens):
            if token in eos:
                sents.append(' '.join(sent_tokens[last_i: i + 1]))
                last_i = i + 1
        if last_i != len(sent_tokens):
            sents.append(' '.join(sent_tokens[last_i: ]))
        return sents

    def create_instances(self, sent: str):
        """
        Convert a sentence into a list of instances.
        """
        sent_tokens = self.model._tokenizer.tokenize(sent)

        # Find all verbs in the input sentence
        pred_ids = [i for (i, t) in enumerate(sent_tokens) if t.pos_ == "VERB"]

        # Create instances
        instances = [{"sentence": sent_tokens, "predicate_index": pred_id} for pred_id in pred_ids]

        return instances

    def get_confidence(self, tag_per_token, class_probs):
        """
        Get the confidence of a given model in a token list, using the class probabilities
        associated with this prediction.
        """
        token_indexes = [self.model._model.vocab.get_token_index(tag, namespace = "labels") for tag in tag_per_token]

        # Get probability per tag
        probs = [class_prob[token_index] for token_index, class_prob in zip(token_indexes, class_probs)]

        # Combine (product)
        prod_prob = functools.reduce(operator.mul, probs)

        return prod_prob

    def get_oie_frame(self, tokens, tags) -> str:
        """
        Converts a list of model outputs (i.e., a list of lists of bio tags, each
        pertaining to a single word), returns an inline bracket representation of
        the prediction.
        """
        frame = defaultdict(list)
        chunk = []
        words = [token.text for token in tokens]

        for (token, tag) in zip(words, tags):
            if tag.startswith("I-") or tag.startswith("B-"):
                frame[tag[2:]].append(token)

        return dict(frame)

    def get_frame_str(self, oie_frame) -> str:
        """
        Convert and oie frame dictionary to string.
        """
        dummy_dict = dict([(k if k != "V" else "ARG01", v)
                        for (k, v) in oie_frame.items()])

        sorted_roles = sorted(dummy_dict)

        frame_str = []
        for role in sorted_roles:
            if role == "ARG01":
                role = "V"
            arg = " ".join(oie_frame[role])
            frame_str.append(f"{role}:{arg}")

        return "\t".join(frame_str)

    def format_extractions(self, sent_tokens, sent_predictions):
        """
        Convert token-level raw predictions to clean extractions.
        """
        # Consolidate predictions
        if not (len(set(map(len, sent_predictions))) == 1):
            raise AssertionError
        assert len(sent_tokens) == len(sent_predictions[0])
        sent_str = " ".join(map(str, sent_tokens))

        pred_dict = consolidate_predictions(sent_predictions, sent_tokens)

        # Build and return output dictionary
        results = []
        all_tags = []

        for tags in pred_dict.values():
            # Join multi-word predicates
            tags = join_mwp(tags)
            all_tags.append(tags)

            # Create description text
            oie_frame = self.get_oie_frame(sent_tokens, tags)

            # Add a predicate prediction to outputs.
            results.append("\t".join([sent_str, self.get_frame_str(oie_frame)]))

        return results, all_tags


class Mock_token:
    """
    Spacy token imitation
    """
    def __init__(self, tok_str):
        self.text = tok_str

    def __str__(self):
        return self.text

if __name__ == '__main__':
    extractor = AllenNLPFactExtractor(cuda=-1)
    while True:
        sent = input()
        facts = extractor.extract(sent)
        # print(extractor.split_sentence(sent))
        print(facts)