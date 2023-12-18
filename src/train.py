

import os
import sys

import json
import random

from typing import List

import fire
import logging

import numpy as np

import torch 
from datasets import Dataset
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    set_seed,
)

from datasets import load_dataset

from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)

from train_utils import Prompter
from ev_my_scores import rouge_score_v2



# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

dtype_map = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

def train(
        # model/data parameters
        model_path                         : str       = "",       # path to the initial model
        train_data_path                    : List[str] = None,     # path to the data
        valid_data_path                    : List[str] = None,     # path to the data
        dataset_sample_limit               : List[int] = None,     # limit the number of samples in each dataset
        with_length_instruction_percentage : int       = 0.0,      # percentage of samples with length instructions
        output_path                        : str       = "",       # path to save the trained models
        dtype_spec                         : str       = "fp32",   # fp32, fp16, or bf16
        model_type                         : str       = "causal", # causal or seq2seq

        # training parameters
        batch_size             : int   = 128,  # batch size (total)
        micro_batch_size       : int   = 4,    # per device batch size
        micro_batch_size_eval  : int   = 8,
        num_epochs             : int   = 3,
        learning_rate          : float = 3e-4,
        warmup_steps           : int   = 100,
        max_input_len          : int   = 1024, # maximum input length
        val_set_size           : int   = 500,
        eval_steps             : int   = 100,
        resume_from_checkpoint : str    = None,
        prompt_template_name   : str   = "",
        logging_steps          : int   = 10,

        # lora hyperparameters
        lora_enabled        : bool      = False,
        lora_r              : int       = 8,
        lora_alpha          : int       = 16,
        lora_dropout        : float     = 0.05,
        lora_target_modules : List[str] = [
            "q_proj",
            "v_proj",
        ],

        # LLM hyperparameters
        train_on_inputs : bool = False,
        add_eos_token   : bool = False,
        group_by_length : bool = False, # can be faster, but produces odd training curve

        # Random seed
        seed : int = 42,
    ):

    set_seed(seed)


    # logging hyperparameters
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info("Training SumLLM model with params:")
        logger.info(f"model_path: {model_path}")
        logger.info(f"train_data_path: {train_data_path}")
        logger.info(f"valid_data_path: {valid_data_path}")
        logger.info(f"dataset_sample_limit: {dataset_sample_limit}")
        logger.info(f"with_length_instruction_percentage: {with_length_instruction_percentage}")
        logger.info(f"output_path: {output_path}")
        logger.info(f"dtype_spec: {dtype_spec}")
        logger.info(f"model_type: {model_type}")

        logger.info(f"batch_size: {batch_size}")
        logger.info(f"micro_batch_size: {micro_batch_size}")
        logger.info(f"micro_batch_size_eval: {micro_batch_size_eval}")
        logger.info(f"num_epochs: {num_epochs}")
        logger.info(f"learning_rate: {learning_rate}")
        logger.info(f"warmup_steps: {warmup_steps}")
        logger.info(f"max_input_len: {max_input_len}")
        logger.info(f"val_set_size: {val_set_size}")
        logger.info(f"eval_steps: {eval_steps}")
        logger.info(f"resume_from_checkpoint: {resume_from_checkpoint}")
        logger.info(f"prompt_template_name: {prompt_template_name}")
        logger.info(f"logging_steps: {logging_steps}")
        logger.info(f"seed: {seed}")

        logger.info(f"lora_enabled: {lora_enabled}")
        if lora_enabled:
            logger.info(f"lora_r: {lora_r}")
            logger.info(f"lora_alpha: {lora_alpha}")
            logger.info(f"lora_dropout: {lora_dropout}")
            logger.info(f"lora_target_modules: {lora_target_modules}")

        if model_type == "causal":
            logger.info(f"train_on_inputs: {train_on_inputs}")
            logger.info(f"add_eos_token: {add_eos_token}")
            logger.info(f"group_by_length: {group_by_length}")

    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        if int(os.environ.get("LOCAL_RANK", 0)) == 0: logger.info("Preparing for DDP...")
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size


    # Load Model
    if int(os.environ.get("LOCAL_RANK", 0)) == 0: 
        logger.info("Loading model...")

    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=dtype_map[dtype_spec],
            device_map=device_map,
            #cache_dir='cache'
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    

    # Load Tokenizer
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    if model_type == 'causal':
        tokenizer.pad_token_id = (0)
        tokenizer.padding_side = "left" # to allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=max_input_len,
            padding=False,
            return_tensors=None,
        )

        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < max_input_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if model_type == "causal":
            result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):

        if model_type == "causal":
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )

            tokenized_full_prompt = tokenize(full_prompt)

            if not train_on_inputs:
                user_prompt = prompter.generate_prompt(
                    data_point["instruction"], data_point["input"]
                )
                tokenized_user_prompt = tokenize(
                    user_prompt, add_eos_token=add_eos_token
                )
                user_prompt_len = len(tokenized_user_prompt["input_ids"])

                if add_eos_token:
                    user_prompt_len -= 1

                tokenized_full_prompt["labels"] = [
                    -100
                ] * user_prompt_len + tokenized_full_prompt["labels"][
                    user_prompt_len:
                ]  # could probably speed up

        elif model_type == "seq2seq":
            full_prompt = prompter.generate_prompt(
                data_point["instruction"], 
                data_point["input"]
            )

            tokenized_full_prompt = tokenize(full_prompt)
            labels = tokenize(data_point["output"], add_eos_token=add_eos_token)
            tokenized_full_prompt['labels'] = labels['input_ids']

        return tokenized_full_prompt

    model.resize_token_embeddings(len(tokenizer)) # no harm to resize embeddings
    
    if lora_enabled:
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info("Preparing model for PEFT (LORA)...")

        if model_type == "causal":
            config = LoraConfig(
                r              = lora_r,
                lora_alpha     = lora_alpha,
                target_modules = lora_target_modules,
                lora_dropout   = lora_dropout,
                bias           = "none",
                task_type      = TaskType.CAUSAL_LM,
            )
        elif model_type == "seq2seq":
            config = LoraConfig(
                r              = lora_r,
                lora_alpha     = lora_alpha,
                target_modules = lora_target_modules,
                lora_dropout   = lora_dropout,
                bias           = "none",
                task_type      = TaskType.SEQ_2_SEQ_LM,
            )

        model = get_peft_model(model, config)
        model.to(dtype_map[dtype_spec])

    # For training data, we load all data from the train_data_path
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info("#### Loading data for training... ####")
    all_train_data = []
    for filename, sample_limit in zip(train_data_path, dataset_sample_limit):
        #with open(filename) as f:
        #    data = json.load(f)
        data = load_dataset("binwang/InstructDS_datasets", filename, split='train')
        new_data = []
        for sample in data:
            new_sample = {
                'instruction': sample['instruction'],
                'input': sample['dialogue'],
                'output': sample['summary'],
            }
            new_data.append(new_sample)
        data = new_data
        
        data = random.sample(data, min(sample_limit, len(data)))
        all_train_data.extend(data)

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info(f"Loaded {len(data)} data points from {filename}")


    # Prepare data
    naive_all_train_data_dict = {
        'instruction': [item['instruction'] for item in all_train_data],
        'input'      : [item['input'] for item in all_train_data],
        'output'     : [item['output'] for item in all_train_data],
    }

    # add length of output to instruction
    subset_train_data = random.sample(all_train_data, int(len(all_train_data) * with_length_instruction_percentage))
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(f"Adding length of output to instruction for {with_length_instruction_percentage*100}% - {len(subset_train_data)} data points")

    with_len_train_data_dict = {
        'instruction': [item['instruction'] + f" The output should be {len(item['output'].split())} words long." for item in subset_train_data],
        'input'      : [item['input'] for item in subset_train_data],
        'output'     : [item['output'] for item in subset_train_data],
    }

    # merge two data dicts
    all_train_data_dict = {
        'instruction': naive_all_train_data_dict['instruction'] + with_len_train_data_dict['instruction'],
        'input'      : naive_all_train_data_dict['input'] + with_len_train_data_dict['input'],
        'output'     : naive_all_train_data_dict['output'] + with_len_train_data_dict['output'],
    }

    raw_train_data = Dataset.from_dict(all_train_data_dict)
    train_data = raw_train_data.shuffle().map(generate_and_tokenize_prompt)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(f"Loaded {len(all_train_data)} data points for training")
        num_truncated = [1 for sample in train_data if len(sample['attention_mask']) >= max_input_len]

        logger.info(f"Number of samples truncated: {sum(num_truncated)}, Percentage: {sum(num_truncated) / len(train_data) * 100}%")
    

    # For validation data, we load all data from the valid_data_path
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info("#### Loading data for validation... ####")
    all_validation_data = []
    for filename in valid_data_path:
        data = load_dataset("binwang/InstructDS_datasets", filename, split='validation')
        new_data = []
        for sample in data:
            new_sample = {
                'instruction': sample['instruction'],
                'input': sample['dialogue'],
                'output': sample['summary'],
            }
            new_data.append(new_sample)
        data = new_data
        all_validation_data.extend(data)

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info(f"Loaded {len(data)} data points from {filename}")
    
    if len(all_validation_data) > val_set_size:
        all_validation_data = random.sample(all_validation_data, val_set_size)

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info(f"Sampled {len(all_validation_data)} data points for validation")
 
    all_validation_data_dict = {
        'instruction': [item['instruction'] for item in all_validation_data],
        'input'      : [item['input'] for item in all_validation_data],
        'output'     : [item['output'] for item in all_validation_data],
    }
    raw_val_data = Dataset.from_dict(all_validation_data_dict)
    val_data = raw_val_data.shuffle().map(generate_and_tokenize_prompt)


    # resume from checkpoint
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    try:
        model.print_trainable_parameters()
    except:
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info("print_trainable_parameters() not implemented, only valid for LoRA models")
        pass

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # data collator
    data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer, model, pad_to_multiple_of=8, return_tensors="pt", padding=True)


    def compute_metrics(eval_pred):

        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge_score = rouge_score_v2(decoded_labels, decoded_preds)

        rouge_score = {k: v['f'] * 100 for k, v in rouge_score.items()}

        return rouge_score


    # training

    if model_type == "seq2seq":
        trainer = transformers.Seq2SeqTrainer(
            model           = model,
            train_dataset   = train_data,
            eval_dataset    = val_data,
            data_collator   = data_collator,
            compute_metrics = compute_metrics,
            args            = transformers.Seq2SeqTrainingArguments(
                per_device_train_batch_size = micro_batch_size ,
                per_device_eval_batch_size  = micro_batch_size_eval ,
                gradient_accumulation_steps = gradient_accumulation_steps ,
                warmup_steps                = warmup_steps ,
                num_train_epochs            = num_epochs ,
                learning_rate               = learning_rate ,
                fp16                        = False if dtype_spec == 'fp16' else False, # known bug with t5
                bf16                        = True if dtype_spec == 'bf16' else False,
                logging_steps               = logging_steps ,
                optim                       = "adamw_torch" ,
                evaluation_strategy         = "steps" if val_set_size > 0 else "no" ,
                save_strategy               = "steps" ,
                eval_steps                  = eval_steps if val_set_size > 0 else None ,
                save_steps                  = eval_steps ,
                log_on_each_node            = False ,
                output_dir                  = output_path ,
                save_total_limit            = 1 ,
                load_best_model_at_end      = True ,
                metric_for_best_model       = "eval_rouge-1" ,
                ddp_find_unused_parameters  = False if ddp else None ,
                group_by_length             = group_by_length ,
                report_to                   = None ,
                run_name                    = None ,
                label_smoothing_factor      = 0.1 , # ask for more input: decoder_input_ids
                predict_with_generate       = True ,
                generation_max_length       = 128 ,
                generation_num_beams        = 4 ,
                #fsdp = "full_shard offload", # fsdp is not compatible with LORA
                #fsdp_transformer_layer_cls_to_wrap="T5Block",
                #deepspeed = "deepspeed_config.json",
                ),
            )



    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.evaluate(val_data)

    # save the final model
    tokenizer.save_pretrained(output_path)
    
    # merge the lora parameters back to the model
    model = model.merge_and_unload()
    model.save_pretrained(output_path)



if __name__ == "__main__":
    fire.Fire(train)