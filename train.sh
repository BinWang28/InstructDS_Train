echo "= = = = = = = = = = = = = ="
echo "The project is running..."
currentDate=`date`
echo $currentDate
start=`date +%s`
echo "= = = = = = = = = = = = = ="

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1

MODEL_INDEX=trial_test

python src/train.py \
    --model_path binwang/flan-t5-xl-own \
    --model_type seq2seq \
    --dtype_spec fp32 \
    --train_data_path "['SAMSum','DialogSum','TODSum','SAMSum_QDS','DialogSum_QDS','TODSum_QDS']" \
    --dataset_sample_limit '[14732,12460,7892,5000,5000,5000]' \
    --with_length_instruction_percentage 1.0 \
    --valid_data_path "['SAMSum','DialogSum']" \
    --output_path trained_model/$MODEL_INDEX \
    --batch_size 128 \
    --micro_batch_size 2 \
    --micro_batch_size_eval 8 \
    --num_epochs 5 \
    --learning_rate 3e-4 \
    --warmup_steps 50 \
    --val_set_size 1000 \
    --eval_steps 120 \
    --max_input_len 768 \
    --prompt_template_name flant5template \
    --logging_steps 5 \
    --lora_enabled True \
    --lora_r 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q,v]' \
    --seed 1234



echo "= = = = = = = = = = = = = ="
echo "The project is Finished..."
end=`date +%s`
runtime=$((end-start))
echo "The program takes '$((runtime / 60))' minutes."
echo "= = = = = = = = = = = = = ="



# bash train.sh 2>&1 | tee train.log
