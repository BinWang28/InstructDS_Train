echo "= = = = = = = = = = = = = ="
echo "The project is running..."
currentDate=`date`
echo $currentDate
start=`date +%s`
echo "= = = = = = = = = = = = = ="


MODEL_INDEX=trial_test

export CUDA_VISIBLE_DEVICES=0


model_path=trained_model/$MODEL_INDEX
lora_weights_path=trained_model/$MODEL_INDEX

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

eval_task=SAMSum
echo "Evaluate data for ###$eval_task### ..."

echo "Inference ..."
python src/ev_inference.py \
    --model_type seq2seq \
    --model_path $model_path \
    --lora_weights_path $lora_weights_path \
    --task $eval_task \
    --prompt_template flant5template \
    --batch_size 5 \
    --max_input_length 1024 \
    --dtype_spec fp32 \


echo "Compute Scores ..."
python src/ev_compute_auto_scores.py \
    --eval_file $lora_weights_path \
    --tasks $eval_task \

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =



eval_task=DialogSum
echo "Evaluate data for ###$eval_task### ..."

echo "Inference ..."
python src/ev_inference.py \
    --model_type seq2seq \
    --model_path $model_path \
    --lora_weights_path $lora_weights_path \
    --task $eval_task \
    --prompt_template flant5template \
    --batch_size 5 \
    --max_input_length 1024 \
    --dtype_spec fp32 \


echo "Compute Scores ..."
python src/ev_compute_auto_scores.py \
    --eval_file $lora_weights_path \
    --tasks $eval_task \

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


eval_task=TODSum
echo "Evaluate data for ###$eval_task### ..."

echo "Inference ..."
python src/ev_inference.py \
    --model_type seq2seq \
    --model_path $model_path \
    --lora_weights_path $lora_weights_path \
    --task $eval_task \
    --prompt_template flant5template \
    --batch_size 5 \
    --max_input_length 1024 \
    --dtype_spec fp32 \


echo "Compute Scores ..."
python src/ev_compute_auto_scores.py \
    --eval_file $lora_weights_path \
    --tasks $eval_task \

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =



eval_task=DREAM
echo "Evaluate data for ###$eval_task### ..."



echo "Inference ..."
python src/ev_inference.py \
    --model_type seq2seq \
    --model_path $model_path \
    --lora_weights_path $lora_weights_path \
    --task $eval_task \
    --prompt_template flant5template \
    --batch_size 5 \
    --max_input_length 1024 \
    --dtype_spec fp32 \



echo "Compute Scores ..."
python src/ev_compute_auto_scores.py \
    --eval_file $lora_weights_path \
    --tasks $eval_task \


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


echo "= = = = = = = = = = = = = ="
echo "The project is Finished..."
end=`date +%s`
runtime=$((end-start))
echo "The program takes '$((runtime / 60))' minutes."
echo "= = = = = = = = = = = = = ="


# bash evaluation.sh 2>&1 | tee evaluation.log
