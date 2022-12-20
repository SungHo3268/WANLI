CUDA=$1
MODEL=$2      # wanli  |  mnli  |  anli


# 'data/wanli/test_sample.jsonl' is 1 batch dummy file.


# big-bench
for t in epistemic_reasoning analytic_entailment disambiguation_qa presuppositions_as_nli
do
	CUDA_VISIBLE_DEVICES=$CUDA bash scripts/finetune_roberta_mine.sh $t data/$MODEL/train.jsonl data/wanli/test_sample.jsonl data/$t/dev.jsonl models/roberta-large-$MODEL
done


for t in diagnostics hans qnli nq-nli fever-nli
do
	CUDA_VISIBLE_DEVICES=$CUDA bash scripts/finetune_roberta_mine.sh $t data/$MODEL/train.jsonl data/wanli/test_sample.jsonl data/$t/dev.jsonl models/roberta-large-$MODEL
done


TASK="wnli"
CUDA_VISIBLE_DEVICES=$CUDA bash scripts/finetune_roberta_mine.sh $TASK data/$MODEL/train.jsonl data/wanli/test_sample.jsonl data/$TASK/train+dev.jsonl models/roberta-large-$MODEL


for t in anli wanli
do
	CUDA_VISIBLE_DEVICES=$CUDA bash scripts/finetune_roberta_mine.sh $t data/$MODEL/train.jsonl data/wanli/test_sample.jsonl data/$t/test.jsonl models/roberta-large-$MODEL
done


TASK="mnli"
for t in dev_matched dev_mismatched
do
	CUDA_VISIBLE_DEVICES=$CUDA bash scripts/finetune_roberta_mine.sh $TASK-$t data/$MODEL/train.jsonl data/wanli/test_sample.jsonl data/$TASK/$t.jsonl models/roberta-large-$MODEL
done
