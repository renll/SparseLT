export G=$1
export GPU=$2
export WAY=$3
export SHOT=$4
export SAVED_MODEL_DIR=$5
export way=${WAY}
export shot=${SHOT}
echo $SAVED_MODEL_DIR
echo $shot $way
export finetune_loss=KL
export is_viterbi=viterbi

MODEL="../../../checkpoints/for_container/"
CONFIG="../../../checkpoints/for_container/"

## training with toy evaluation for sanity check
python src/container.py --data_dir data/few-nerd/${G} --labels-train data/few-nerd/${G}/labels_train.txt --labels-test data/few-nerd/${G}/labels_test.txt --config_name $CONFIG --model_name_or_path $MODEL --saved_model_dir saved_models/few-nerd/${G}/${SAVED_MODEL_DIR} --output_dir outputs/few-nerd/${G}/${finetune_loss}_${is_viterbi}_final_5000_${SAVED_MODEL_DIR}/${G}-${way}-${shot}/ --support_path support_test_${way}_${shot}/ --test_path query_test_${way}_${shot}/ --n_shots ${shot} --max_seq_length 128 --embedding_dimension 128 --num_train_epochs 1 --train_batch_size 32 --seed 1 --do_train --do_predict --select_gpu ${GPU} --training_loss KL --finetune_loss ${finetune_loss} --evaluation_criteria euclidean_hidden_state --consider_mutual_O --learning_rate 1e-4 --learning_rate_finetuning 1e-4

## evaluation
echo $shot $way 
python src/container.py --data_dir data/few-nerd/${G} --labels-train data/few-nerd/${G}/labels_train.txt --labels-test data/few-nerd/${G}/labels_test.txt --config_name $CONFIG --model_name_or_path $MODEL --saved_model_dir saved_models/few-nerd/${G}/${SAVED_MODEL_DIR} --output_dir outputs/few-nerd/${G}/${finetune_loss}_${is_viterbi}_final_5000_${SAVED_MODEL_DIR}/${G}-${way}-${shot}/ --support_path support_test_${way}_${shot}/ --test_path query_test_${way}_${shot}/ --n_shots ${shot} --max_seq_length 128 --embedding_dimension 128 --num_train_epochs 1 --train_batch_size 32 --seed 1 --do_predict --select_gpu ${GPU} --training_loss KL --finetune_loss ${finetune_loss} --evaluation_criteria euclidean_hidden_state --learning_rate 1e-4 --learning_rate_finetuning 1e-4 --consider_mutual_O --temp_trans 0.01 --silent
