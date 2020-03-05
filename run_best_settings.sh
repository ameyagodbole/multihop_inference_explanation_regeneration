TRAIN_QUESTIONS_FILE="questions/ARC-Elementary+EXPL-Train.tsv"
DEV_QUESTIONS_FILE="questions/ARC-Elementary+EXPL-Dev.tsv"
TEST_QUESTIONS_FILE="questions/ARC-Elementary+EXPL-Test-Masked.tsv"
FACTS_FILE="questions/all_facts.tsv"
FACT_ADJ_LIST_FILE="fact_graph/fact_as_node/adjacency_map.pkl"
FACT_FREQ_FILE="./questions/fact_usage_frequency.pkl"

# Intermediate examples and features files will be cached here
# Example files are necessary for generating final predictions
DATA_DIR="./questions"

# Prediction files
ENSEMBLE_PRED_FILE="./predictions/predict-bert-test-path-rank-1e-k50-rerank-3e-sl140-ensemble.txt"
FINAL_PRED_FILE="./predictions/predict-bert-test-path-rank-1e-k50-rerank-3e-sl140-ensemble-move-redundant.txt"

RERANK_TRAIN_SEQ=72
RERANK_PRED_SEQ=140
RERANK_OUTPUT_DIR="./outputs/bert_rerank_correctchoices_unweighted/"

PATH_RANK_TRAIN_SEQ=90
PATH_RANK_TRAIN_TFIDF=25
PATH_RANK_PRED_SEQ=140
PATH_RANK_PRED_TFIDF=50
PATH_RANK_OUTPUT_DIR="./outputs/bert_path_rank_correctchoices_unweighted_k25_1e/"

# Train bert-reranker
python -u bert_reranker.py --model_type bert --model_name_or_path bert-base-uncased \
  --task_name TG2019_reranker_correct \
  --do_train --do_eval \
  --do_lower_case \
  --data_dir ${DATA_DIR} \
  --max_seq_length ${RERANK_TRAIN_SEQ} \
  --per_gpu_eval_batch_size=64 \
  --per_gpu_train_batch_size=64 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --logging_steps 5000 \
  --evaluate_during_training --evaluation_steps 25000 --save_steps 5000 \
  --output_dir ${RERANK_OUTPUT_DIR} \
  --train_questions_file ${TRAIN_QUESTIONS_FILE} \
  --dev_questions_file ${DEV_QUESTIONS_FILE} \
  --facts_file ${FACTS_FILE} --mcq_choices correct

# Get predictions from bert-reranker (base predictions)
python -u bert_reranker.py --model_type bert --model_name_or_path bert-base-uncased \
  --task_name TG2019_reranker_correct \
  --do_predict \
  --do_lower_case \
  --data_dir ${DATA_DIR} \
  --max_seq_length ${RERANK_PRED_SEQ} \
  --output_dir ${RERANK_OUTPUT_DIR} \
  --test_questions_file ${TEST_QUESTIONS_FILE} \
  --facts_file ${FACTS_FILE} --mcq_choices correct

# Train bert-path-ranker
python -u bert_path_ranker.py --model_type bert --model_name_or_path bert-base-uncased \
  --task_name TG2019_path_ranker_correct \
  --do_train --do_eval \
  --evaluate_during_training \
  --do_lower_case \
  --data_dir ${DATA_DIR} \
  --max_seq_length ${PATH_RANK_TRAIN_SEQ} \
  --per_gpu_eval_batch_size 50 \
  --per_gpu_train_batch_size 50 \
  --learning_rate 2e-5 \
  --num_train_epochs 1.0 \
  --logging_steps 5000 \
  --save_steps 5000 \
  --evaluation_steps 50000 \
  --output_dir ${PATH_RANK_OUTPUT_DIR} \
  --train_questions_file ${TRAIN_QUESTIONS_FILE} \
  --dev_questions_file ${DEV_QUESTIONS_FILE} \
  --facts_file ${FACTS_FILE} \
  --mcq_choices correct \
  --adjacency_list_file ${FACT_ADJ_LIST_FILE} \
  --tfidf_top_k ${PATH_RANK_TRAIN_TFIDF}

# Get predictions from bert-path-ranker (overlay predictions)
python -u bert_path_ranker.py --model_type bert --model_name_or_path bert-base-uncased \
  --task_name TG2019_path_ranker_correct \
  --do_predict \
  --do_lower_case \
  --data_dir ${DATA_DIR} \
  --max_seq_length ${PATH_RANK_PRED_SEQ} \
  --per_gpu_eval_batch_size 80 \
  --per_gpu_train_batch_size 80 \
  --output_dir ${PATH_RANK_OUTPUT_DIR} \
  --test_questions_file ${TEST_QUESTIONS_FILE} \
  --facts_file ${FACTS_FILE} \
  --mcq_choices correct \
  --adjacency_list_file ${FACT_ADJ_LIST_FILE} \
  --tfidf_top_k ${PATH_RANK_PRED_TFIDF}

# Combine predictions and create submission
python -u ensemble.py \
  --questions_file ${TEST_QUESTIONS_FILE} \
  --facts_file ${FACTS_FILE} \
  --overlay_examples_file ${DATA_DIR}/bert_path2_rerank_cached_examples_test_bert-base-uncased_${PATH_RANK_PRED_SEQ}_${PATH_RANK_PRED_TFIDF}_tg2019_path_ranker_correct \
  --overlay_logits_file ${PATH_RANK_OUTPUT_DIR}/test_preds.npy \
  --base_examples_file ${DATA_DIR}/bert_rerank_cached_examples_test_bert-base-uncased_${RERANK_PRED_SEQ}_tg2019_correct \
  --base_logits_file ${RERANK_OUTPUT_DIR}/test_preds.npy \
  --pred_output_file ${ENSEMBLE_PRED_FILE} \
  --ensemble

python -u ensemble.py \
  --questions_file ${TEST_QUESTIONS_FILE} \
  --facts_file ${FACTS_FILE} \
  --fact_frequency_file ${FACT_FREQ_FILE} \
  --predictions_file ${ENSEMBLE_PRED_FILE} \
  --pred_output_file ${FINAL_PRED_FILE} \
  --move_redundant
