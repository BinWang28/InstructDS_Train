#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Wednesday, May 3rd 2023, 12:47:43 pm
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

import fire
import logging

from tqdm import trange

import torch

from peft import PeftModel

from transformers import (
    GenerationConfig, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM
)
from datasets import load_dataset

from train_utils import Prompter


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def main(
    model_type       : str = "",
    model_path       : str = "",
    lora_weights_path: str = "",
    prompt_template  : str = "",
    task             : str = "",
    batch_size       : int = 4,
    max_input_length : int = 1024,
    dtype_spec       : str = "fp32",
    )                : 

    prompter = Prompter(prompt_template)

    device_map = "auto"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    lora_enabled = not (lora_weights_path == model_path)

    if model_type == "seq2seq": # e.g. T5
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, 
            torch_dtype=dtype_map[dtype_spec],
            device_map=device_map,
        )
        if lora_enabled:
            print('Loading LORA weights from: ' + lora_weights_path)
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                torch_dtype=dtype_map[dtype_spec],
            )

    else:
        raise ValueError("Model type must be one of 'seq2seq' or 'causal'")


    if torch.__version__ >= "2":
        model = torch.compile(model)

    def evaluate(
        instructions,
        inputs           = None,
        temperature      = 1.0,
        top_p            = 1.0,
        top_k            = 50,
        num_beams        = 4,
        max_new_tokens   = 128,
        infer_batch_size = batch_size,
        **kwargs,
    ):
        prompts = [prompter.generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]

        generation_config = GenerationConfig(
            #temperature = temperature,
            #top_p       = top_p,
            #top_k       = top_k,
            num_beams   = num_beams,
            **kwargs,
        )

        all_responses = []

        for i in trange(0, len(prompts), infer_batch_size):
            batch_prompts = prompts[i : i + infer_batch_size]

            tokenized_inputs = tokenizer(
                batch_prompts, 
                return_tensors = "pt",
                padding        = True,
                truncation     = True,
                max_length     = max_input_length,
                ).to(model.device)

            with torch.no_grad():
                batch_output = model.generate(
                    **tokenized_inputs,
                    generation_config       = generation_config,
                    return_dict_in_generate = True,
                    output_scores           = True,
                    max_new_tokens          = max_new_tokens,
                )

            batch_seq = tokenizer.batch_decode(batch_output[0], skip_special_tokens=True)

            generated_responses = [prompter.get_response(output) for output in batch_seq]
            generated_responses = [response.split(tokenizer.eos_token)[0] for response in generated_responses]

            all_responses.extend(generated_responses)

        return all_responses
    


    # Load data
    data = load_dataset("binwang/InstructDS_datasets", task, split='test')
    
    if task != "DREAM":
        new_data = []
        for sample in data:
            new_sample = {
                'instruction': sample['instruction'],
                'input'      : sample['dialogue'],
                'output'     : sample['summary'],
            }
            new_data.append(new_sample)
        data = new_data

    else:
        new_data = []
        for sample in data:
            new_sample = {
                'instruction': sample['instruction'] + " " + sample['question'],
                'input'      : sample['dialogue'],
                'output'     : sample['summary'],
                'choices'    : sample['choices'],
            }
            new_data.append(new_sample)
        data = new_data
    
    # Inference
    instructions = [sample['instruction'] for sample in data]
    inputs = [sample['input'] for sample in data]
    outputs = [sample['output'] for sample in data]

    generated_responses = evaluate(instructions, inputs)

    data_with_generated_response = []
    for i in range(len(instructions)):

        if task == "DREAM":
            new_sample = {
                'instruction'       : instructions[i],
                'input'             : inputs[i],
                'output'            : outputs[i],
                'generated_response': generated_responses[i],
                'choices'           : data[i]['choices'],
            }
        
        else:
            new_sample = {
                'instruction'       : instructions[i],
                'input'             : inputs[i],
                'output'            : outputs[i],
                'generated_response': generated_responses[i],
            }

        data_with_generated_response.append(new_sample)

    os.makedirs(lora_weights_path, exist_ok=True)
    with open(os.path.join(lora_weights_path, task+'_gen.json'), 'w') as f:
        json.dump(data_with_generated_response, f, indent=4)


if __name__ == "__main__":
    fire.Fire(main)