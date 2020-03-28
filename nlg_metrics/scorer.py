from nlg_metrics.rouge.rouge import RougeScorer as ROUGE
from nlg_metrics.bertscore.bert_util import bert_cos_score_idf, get_model, model2layers, cache_scibert
from nlg_metrics.factscore.fact_extractor import AllenNLPFactExtractor, KnowItAllFactExtractor
from nlg_metrics.factscore.sent_encoder import InferSentSentenceEncoder, GoogleUniversalSentenceEncoder
from nlg_metrics.factscore.fact_connector import BasicFactConnector
from nlg_metrics.factscore.score_calculator import DotProductScoreCalculator, DotProductWithThresholdScoreCalculator
from nlg_metrics.factscore.plot_util import plot_heatmap
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer
from typing import List
import re
import numpy as np
import json
import torch

class Scorer(object):
    '''
    # Usage:
    from scorer import Scorer
    scorer = Scorer()
    score = scorer.score(gens, refs)
    # score ranges from 0 to 100
    '''
    def __init__(self, name: str):
        self.name = name

    def score(self, gens: List[List[str]], refs: List[List[str]]) -> List[float]:
        raise NotImplementedError()

class RougeScorer(Scorer):
    def __init__(self, metric='rougeLsum', tags=['<t>', '</t>'], stem=True):
        super(RougeScorer, self).__init__(name='RougeScorer')
        self.metric = metric
        self.stem = stem
        self.tags = tags
        self.scorer = ROUGE(metrics=[self.metric], stem=self.stem)
        
    def score(self, gens: List[str], refs: List[str]) -> List[float]:
        assert(len(gens) == len(refs))
        gens, refs = self.preprocess(gens, refs)
        results = []
        for gen, ref in zip(gens, refs):
            s = self.scorer.score_each(gen, ref)
            results.append(s)
        return results

    def split_sentences(self, article: str) -> List[str]:
        # bare_sents = re.findall(r'%s (.+?) %s' % (self.tags[0], self.tags[1]), article)
        bare_sents = [article]
        return bare_sents

    def preprocess(self, gens: List[List[str]], refs: List[List[str]]):
        for i in range(len(gens)):
            gens[i] = '\n'.join(self.split_sentences(gens[i]))
            refs[i] = '\n'.join(self.split_sentences(refs[i]))
        return gens, refs

class BertScorer(Scorer):
    def __init__(self, model_type='bert-base-uncased', batch_size=128, tags=['<t>', '</t>']):
        super(BertScorer, self).__init__(name='BertScorer')
        self.tags = tags
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_type = model_type
        self.batch_size = batch_size
        self.model = None
        if model_type.startswith('scibert'):
            self.tokenizer = AutoTokenizer.from_pretrained(cache_scibert(self.model_type))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)

        self.idf_dict = defaultdict(lambda: 1.)
        self.idf_dict[self.tokenizer.sep_token_id] = 0
        self.idf_dict[self.tokenizer.cls_token_id] = 0
        self.num_layers = model2layers[model_type]

    def bert_score_load_model(self):
        self.model = get_model(self.model_type, self.num_layers, all_layers=False)
        self.model.to(self.device)
        return

    def score(self, gens: List[str], refs: List[str], verbose=False) -> List[float]:
        assert len(gens) == len(refs)
        if self.model is None:
            self.bert_score_load_model()
        gens, refs = self.preprocess(gens, refs)
        all_preds = bert_cos_score_idf(self.model, refs, gens, self.tokenizer, self.idf_dict, verbose=verbose, device=self.device, batch_size=self.batch_size, all_layers=False)
        F1 = all_preds[..., 2].cpu().numpy().tolist()
        F1 = [100 * i for i in F1]
        return F1

    def remove_tags(self, sentence):
        for tag in self.tags:
            sentence = sentence.replace(tag, '')
        sentence = ' '.join(sentence.split())
        return sentence

    def preprocess(self, gens: List[List[str]], refs: List[List[str]]):
        for i in range(len(gens)):
            gens[i] = self.remove_tags(gens[i])
            refs[i] = self.remove_tags(refs[i])
        return gens, refs


class FactScorer(Scorer):
    def __init__(self, tags=['<t>', '</t>'], encoder_type='GoogleUniversalSentenceEncoder', extractor_type='KnowItAllFactExtractor', scorer_type='DotProductWithThresholdScoreCalculator', threshold=0):
        super(FactScorer, self).__init__(name='FactScorer')
        self.encoder_type = encoder_type
        self.extractor_type = extractor_type
        self.scorer_type = scorer_type
        self.threshold = threshold
        self.tags = tags
        
    @property
    def extractor(self):
        if not hasattr(self, '_extractor'):
            if self.extractor_type == 'KnowItAllFactExtractor':
                self._extractor = KnowItAllFactExtractor()
            elif self.extractor_type == 'AllenNLPFactExtractor': 
                self._extractor = AllenNLPFactExtractor()
        return self._extractor

    @property
    def encoder(self):
        if not hasattr(self, '_encoder'):
            if self.encoder_type == 'GoogleUniversalSentenceEncoder':
                self._encoder = GoogleUniversalSentenceEncoder()
            elif self.encoder_type == 'InferSentSentenceEncoder':
                self._encoder = InferSentSentenceEncoder()
        return self._encoder

    @property
    def connector(self):
        if not hasattr(self, '_connector'):
            self._connector = BasicFactConnector()
        return self._connector

    @property
    def calculator(self):
        if not hasattr(self, '_calculator'):
            if self.scorer_type == 'DotProductWithThresholdScoreCalculator':
                self._calculator = DotProductWithThresholdScoreCalculator(self.threshold)
            else:
                self._calculator = DotProductScoreCalculator()
        return self._calculator
        
    def score(self, gens: List[str], refs: List[str], return_all=False) -> List[float]:
        gens_facts = self.extractor.extract_list(gens)
        refs_facts = self.extractor.extract_list(refs)
        gens_sents = [self.connector.connect_list(facts) for facts in gens_facts]
        refs_sents = [self.connector.connect_list(facts) for facts in refs_facts]
        gens_embs = [self.encoder.encode_list(sents) for sents in gens_sents]
        refs_embs = [self.encoder.encode_list(sents) for sents in refs_sents]
        scores = self.calculator.calculate_list(gens_embs, refs_embs)
        if return_all:
            return (gens_facts, refs_facts, gens_sents, refs_sents, gens_embs, refs_embs, scores)
        return [100 * score['f1'] for score in scores]

    def score_each(self, gen: str, ref: str, return_all=False, verbose=False, heatmap=False) -> float:
        assert(type(gen) is str and type(ref) is str)
        gen_facts, ref_facts, gen_sents, ref_sents, gen_embs, ref_embs, score = self.score([gen], [ref], return_all=True)
        gen_facts, ref_facts, gen_sents, ref_sents, gen_embs, ref_embs, score = gen_facts[0], ref_facts[0], gen_sents[0], ref_sents[0], gen_embs[0], ref_embs[0], score[0]
        if verbose:
            print('Generated Facts:')
            self.pretty_print(gen_facts, prefix='GEN')
            print('Reference Facts:')
            self.pretty_print(ref_facts, prefix='REF')
        if heatmap:
            mat = score['mat']
            if mat is not None:
                v, h = mat.shape
                v_names, h_names = [f'G{i}' for i in range(v)], [f'R{i}' for i in range(h)]
                plot_heatmap(h_names, v_names, mat * 100)
        if return_all:
            return gen_facts, ref_facts, gen_sents, ref_sents, gen_embs, ref_embs, score
        return 100 * score['f1']

    def pretty_print(self, facts: List[List[str]], prefix='R'):
        print('-' * 63)
        for i, fact in enumerate(facts):
            print(f'{prefix}{i}: {fact}')
        print('-' * 63)


if __name__ == '__main__':
    def aeq(src, tgt, epsilon=1e-3):
        assert(type(src) == type(tgt)), f'type {type(src)} != type {type(tgt)}'
        if type(src) is list: assert(all(-epsilon < i - j < epsilon for i, j in zip(src, tgt))), f'{src} != {tgt}'
        else: assert(-epsilon < src - tgt < epsilon), f'{src} != {tgt}'

    # Test Case #1 for ROUGE
    scorer = RougeScorer()
    scores = scorer.score(['<t> hello world </t>'], ['<t> hello world </t>'])
    aeq(scores, [100])
    print('Test case #1 passed...')
    # Test Case #2 for ROUGE
    reference1 = "<t> marseille prosecutor says `` so far no videos were used in the crash investigation '' despite media reports . </t> <t> journalists at bild and paris match are `` very confident '' the video clip is real , an editor says . </t> <t> andreas lubitz had informed his lufthansa training school of an episode of severe depression , airline says . </t>\n"
    generated1 = "<t> marseille prosecutor brice robin told cnn `` so far no videos were used in the crash investigation '' </t> <t> robin 's comments follow claims by two magazines , german daily bild and french paris match . </t> <t> paris match and bild reported that the video was recovered from a phone at the wreckage site . </t>\n"
    reference2 = "<t> membership gives the icc jurisdiction over alleged crimes committed in palestinian territories since last june . </t> <t> israel and the united states opposed the move , which could open the door to war crimes investigations against israelis . </t>\n"
    generated2 = "<t> the palestinian authority officially became the 123rd member of the international criminal court on wednesday . </t> <t> the icc opened a preliminary examination into the situation in palestinian territories , paving the way for possible war crimes investigations against israelis . </t> <t> but palestinian foreign minister riad al-malki said it was a move toward greater justice . </t>\n"
    reference3 = "<t> amnesty 's annual death penalty report catalogs encouraging signs , but setbacks in numbers of those sentenced to death . </t> <t> organization claims that governments around the world are using the threat of terrorism to advance executions . </t> <t> the number of executions worldwide has gone down by almost 22 % compared with 2013 , but death sentences up by 28 % . </t>\n"
    generated3 =  "<t> `` the dark trend of governments using the death penalty in a futile attempt to tackle real or imaginary threats to state security and public safety was stark last year , '' amnesty international says . </t> <t> at least 607 people were executed around the world in 2014 , compared to 778 in 2013 . </t> <t> the number of executions worldwide has gone down by almost 22 % on the previous year . </t>\n"
    scores = scorer.score([generated1, generated2, generated3], [reference1, reference2, reference3])
    aeq(scores, [37.11340206185568, 32.558139534883715, 46.15384615384615])
    print('Test case #2 passed...')
    # Test Case #1 for BERT
    scorer = BertScorer()
    scores = scorer.score(['hello world'], ['hello world'])
    aeq(scores, [100])
    print('Test case #1 passed...')
    # Test Case #2 for BERT
    scores = scorer.score([generated1, generated2, generated3], [reference1, reference2, reference3])
    aeq(scores, [68.05665493011475, 64.4581139087677, 66.181081533432])
    print('Test case #2 passed...')
    # Test Case #1 for FACT
    scorer = FactScorer()
    generated1 = "Police are searching for a man charged with killing his wife at the Dunkin ' Donuts in Maryland. Police said that \u2583-year-old Bhadreshkumar Chetanbhai Patel, hit his wife Palak Bhadreskumar Patel multiple times with an object and then fled Anne Arundel County police said a severely injured woman in the kitchen area. The woman, \u2583, and Palak Bhadreskumar Patel of Hanover, died at the scene, police said in a news release said that Palak Bhadreskumar Patel has been charged with first-degree murder, second-degree assault."
    reference1 = "Patrons called police sunday night shortly after discovering no workers at the Dunkin ' Donuts on Arundel Mills Boulevard in Hanover. According to authorities, an officer checked the business and found a severely injured woman in the kitchen area. Police said Palak Bhadreskumar Patel of Hanover, died at the scene. They said the investigation revealed that her husband, Bhadreshkumar Chetanbhai Patel, hit her multiple times with an object and then fled."
    generated2 = "Xu, known only as Xu, even tried to impregnate the 'daughter-in-law' himself after discovering his son had not slept with his wife. He bought a wife for his son, then reselling her to another family. Xu is currently under arrest for human trafficking by the police of Lianyungang in eastern China. He spent \u2583,\u2583 yuan yuan (\u00e2\u00a3 \u2583,\u2583) and buying a wife. Xu. Xu was to have sex with her himself. Xu, who has learning difficulties and his mental health was."
    reference2 = "The man, known only as Xu, was keen to have a grandchild. He bought the 'daughter-in-law' for \u00e2\u00a3 0,000 at the beginning of 0000. Both his son and the woman that Xu bought suffered from learning difficulties. After realising the young couple were not sleeping together, he began having sex with her himself. But when she failed to fall pregnant he decided to sell her on again. Xu is under arrest for human trafficking at Lianyungang in eastern China."
    generated3 = "Jos hooiveld's late strike earned millwall a crucial 2-1 victory over south London rivals Charlton on Good Friday - The lions' first win at the den since October. hooiveld prodded home Magaye Gueye's low drive in the 87th minute, after Gueye had cancelled out Alou Diarra's strike for the addicks. Charlton had skipper Chris Solly dismissed midway through the first half for deliberate handball in the box, but Stephen Henderson saved Lee Gregory's resulting penalty."
    reference3 = "Millwall beat 00-men Charlton 0-0 in their championship clash at the den. Addicks defender Chris Solly was sent off for a handball in the area. Stephen Henderson saved the subsequent penalty from Lee Gregory. Alou Diarra scored for visitors against the run of play in the second half. Magaye Gueye levelled for the Lions before Jos Hooiveld netted late winner."
    generated4 = "Ocean spray cranberry classic juice drink was found to have 11g of sugar per 100ml. Children under ten get almost a fifth of their sugar intake from soft drinks. Some cans of fizzy drinks contain almost twice the recommended daily sugar limit."
    reference4 = "Fruit juices made from so-called superfoods such as cranberries and pomegranates are lauded for health benefits. But ome brands contain more than a day âs recommended intake of sugar in a single 000ml serving. Local Government Association accused soft drink firms of â€˜ dragging their heels â€™ when it comes to minimising sugar in their products. It said children under ten get almost a fifth of sugar intake from soft drinks."
    scores = scorer.score([reference1], [reference1])
    aeq(scores, [100])
    print('Test case #1 passed...')
    # Test Case #2 for FACT
    scores = scorer.score([generated1, generated2], [reference1, reference2])
    aeq(scores, [50.61199739601553, 60.67722085368829])
    print('Test case #2 passed...')
    # Test Case #3 for FACT
    scorer.score_each(generated4, reference4, verbose=True, heatmap=True)

