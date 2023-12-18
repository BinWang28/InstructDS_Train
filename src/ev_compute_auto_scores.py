#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Friday, April 28th 2023, 4:04:14 pm
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###

import os
import json
import logging
import argparse

from tqdm import tqdm
import numpy as np

import bert_score

from ev_my_scores import rouge_score_v1, rouge_score_v2, bertscore_score


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


def main():
    '''main'''

    # arguments
    args = parse_args()

    # get tasks
    task_names = [item for item in args.tasks.split(',') if len(item) > 1]

    for task_name in task_names:
        # load test data
        with open(os.path.join(args.eval_file, task_name + '_gen.json'), 'r') as f:
            data = json.load(f)

        ref_results = []
        gen_results = []

        for sample in tqdm(data):
            ref_results.append(sample['output'])
            gen_results.append(sample['generated_response'])
        
        logger.info("Evaluation on {}".format(task_names))
        
        if task_name != 'DREAM':
        
            logger.info("")
            logger.info("Computing ROUGE V1 scores...")
            rouge_score_v1(ref_results, gen_results)

            logger.info("")
            logger.info("Computing ROUGE V2 scores...")
            rouge_score_v2(ref_results, gen_results)

            logger.info("")
            logger.info("Computing BERTScore...")
            bertscore_score(ref_results, gen_results)

        elif task_name == 'DREAM':

           
            all_choices = [sample['choices'] for sample in data]
            choices_list = sum(all_choices, [])

            all_responses = [[gen_response] * 3 for gen_response in gen_results]
            responses_list = sum(all_responses, [])


            scores1 = bert_score.score(responses_list, choices_list.copy(), lang='en', rescale_with_baseline=True)[2]


            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scorer.score(responses_list[0], choices_list.copy()[0])

            scores2 = [scorer.score(item1, item2)['rougeL'][-1] for item1, item2 in zip(responses_list,choices_list.copy())]

            correct_count = 0
            choices = choices_list.copy()
            for i in range(0, len(scores1), 3):
                index = np.argmax(scores1[i:i+3])
                chosen_answer = choices[i:i+3][index]
                if chosen_answer == ref_results[i//3]:
                    correct_count += 1
            print("Accuracy from BERTScore: ", correct_count/len(gen_results))

            correct_count = 0
            choices = choices_list.copy()
            for i in range(0, len(scores2), 3):
                index = np.argmax(scores2[i:i+3])
                chosen_answer = choices[i:i+3][index]
                if chosen_answer == ref_results[i//3]:
                    correct_count += 1
            print("Accuracy from ROUGE-L: ", correct_count/len(gen_results))

            

def parse_args():
    '''parse args'''

    parser = argparse.ArgumentParser(description='Evaluate the generated summaries')

    parser.add_argument(
        '--eval_file', 
        type=str, 
        default=None, 
        help='The directory of the model and outputs to be evaluated.'
        )
    
    parser.add_argument(
        '--tasks',
        type=str,
        default=None,
        help="The tasks to be evaluated, separated by comma."
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()