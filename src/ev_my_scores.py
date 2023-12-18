#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Monday, April 17th 2023, 4:07:38 pm
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
# 2023-04-17 16:14:31	B.W	ROUGE SCORE V2 = implementation in huggingface
# 2023-04-17 16:14:07	B.W	ROUGE SCORE V1 = py-rouge=1.1
###

import os
import logging
import numpy as np

import rouge
from rouge_score import rouge_scorer

import bert_score

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

def prepare_results(m, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(m, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

def rouge_score_v1(references, generations):
    ''' https://pypi.org/project/py-rouge/ '''

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                            max_n=2,
                            limit_length=False,
                            apply_avg=True,
                            apply_best=False,
                            alpha=0.5, # Default for F1_score
                            weight_factor=1.2, # Official default for Rouge-W
                            stemming=True)

    scores = evaluator.get_scores(generations, references)

    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        logger.info(prepare_results(metric, results['p'], results['r'], results['f']))

    return scores


def rouge_score_v2(referemces, generations):
    ''' Huggingface / Google implementation '''

    #scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    #scores = scorer.score('The quick brown fox jumps over the lazy dog',
    #                  'The quick brown dog jumps on the log.')


    #results = rouge.compute(predictions=generations, references=referemces)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores = {
        'rouge1': {
            'precision': [],
            'recall': [],
            'fmeasure': [],
        },
        'rouge2': {
            'precision': [],
            'recall': [],
            'fmeasure': [],
        },
        'rougeL': {
            'precision': [],
            'recall': [],
            'fmeasure': [],
        },
    }

    # for each of the hypothesis and reference documents pair
    for (ref, gen) in zip(referemces, generations):
        # computing the ROUGE
        score = scorer.score(ref, gen)

        # separating the measurements
        precision, recall, fmeasure = score['rouge1']
        scores['rouge1']['precision'].append(precision)
        scores['rouge1']['recall'].append(recall)
        scores['rouge1']['fmeasure'].append(fmeasure)

        precision, recall, fmeasure = score['rouge2']
        scores['rouge2']['precision'].append(precision)
        scores['rouge2']['recall'].append(recall)
        scores['rouge2']['fmeasure'].append(fmeasure) 
        
        precision, recall, fmeasure = score['rougeL']
        scores['rougeL']['precision'].append(precision)
        scores['rougeL']['recall'].append(recall)
        scores['rougeL']['fmeasure'].append(fmeasure) 

    avg_scores = {
        'rouge-1': {
            'p': np.mean(scores['rouge1']['precision']),
            'r': np.mean(scores['rouge1']['recall']),
            'f': np.mean(scores['rouge1']['fmeasure']),
        },
        'rouge-2': {
            'p': np.mean(scores['rouge2']['precision']),
            'r': np.mean(scores['rouge2']['recall']),
            'f': np.mean(scores['rouge2']['fmeasure']),
        },
        'rouge-l': {
            'p': np.mean(scores['rougeL']['precision']),
            'r': np.mean(scores['rougeL']['recall']),
            'f': np.mean(scores['rougeL']['fmeasure']),
        },
    }

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        for metric, results in sorted(avg_scores.items(), key=lambda x: x[0]):
            logger.info(prepare_results(metric, results['p'], results['r'], results['f']))

    return avg_scores


def bertscore_score(references, generations):
    ''' A proxy for BERTScore '''

    logger.info("The BERTScore version: {}".format(bert_score.__version__))

    P, R, F1 = bert_score.score(generations, references, lang='en', verbose=False, rescale_with_baseline=True)

    scores = {
        'precision': P.mean(),
        'recall': R.mean(),
        'fmeasure': F1.mean(),
    }

    logger.info("BERTScore - Prevision: {:.2f}, Recall: {:.2f}, F-1 measure: {:.2f}".format(scores['precision']*100, scores['recall']*100, scores['fmeasure']*100))

    return scores
