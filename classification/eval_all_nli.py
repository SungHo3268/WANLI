import os
import json
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--host', type=str, default='dummy')
parser.add_argument('--port', type=int, default=56789)
parser.add_argument('--model_path', type=str, default="models/roberta-large-mnli")
parser.add_argument('--task_name', type=str, default="wanli", help="all  |  diagnostics  |  hans  |  qnli  |  wnli  |  nq-nli  |  "
                                                                   "anli  |  fever-nli  |  big-bench  |  wanli  |  mnli")
args = parser.parse_args()

pred_dir = os.path.join(args.model_path, "predictions")

print("\n###################################################")
for task_name in ["diagnostics", "hans", "qnli", "wnli", "nq-nli", "anli", "fever-nli", "big-bench", "wanli", "mnli"]:
    data_dir = os.path.join(f"data/{task_name}/")
    data_file = None
    if task_name == 'wnli':
        data_file = os.path.join(data_dir, "train+dev.jsonl")
    elif task_name in ['anli', 'wanli']:
        data_file = os.path.join(data_dir, "test.jsonl")
    elif task_name == 'mnli':
        data_file = [os.path.join(data_dir, "dev_matched.jsonl"), os.path.join(data_dir, "dev_mismatched.jsonl")]
    elif task_name == 'big-bench':
        data_file = [os.path.join(f"data/{task}", "dev.jsonl") for task in ["epistemic_reasoning", "analytic_entailment", "disambiguation_qa", "presuppositions_as_nli"]]
    else:   # ['diagnostics', 'hans', 'qnli', 'nq-nli', 'fever-nli']
        data_file = os.path.join(data_dir, "dev.jsonl")
    
    
    print(f"task_name: {task_name}")
    print(f"data_file: {data_file}")
    
    
    if task_name == 'mnli':
        print("-------------------------------------------------")
        for d_file in data_file:
            predictions = open(os.path.join(pred_dir, f"{task_name}-{os.path.basename(d_file).split('.')[0]}_predictions.txt"), "r").readlines()
            predictions = [line.strip().split('\t')[1] for line in predictions[1:]]

            original = open(d_file, "r").readlines()
            original_data = []
            for line in original:
                line = json.loads(line)
                original_data.append(line['gold'])

            if len(predictions) != len(original_data):
                original_data = original_data[:len(predictions)]
            
            a = np.where(np.array(predictions) == np.array(original_data))[0]
            acc = len(a) / len(original_data) * 100
            print(f"Prediction acc on {task_name}-{os.path.basename(d_file)}: {acc:.2f}")
            
    elif task_name == 'big-bench':
        total_results = []
        task_acc = {}
        for d_file in data_file:
            task = d_file.split('/')[1]
            predictions = open(os.path.join(pred_dir, f"{task}_predictions.txt"), "r").readlines()
            predictions = [line.strip().split('\t')[1] for line in predictions[1:]]
            if task in ['epistemic_reasoning', "analytic_entailment"]:
                label_mapping = {"entailment": "entailment",
                                 "neutral": "non-entailment",
                                 "contradiction": "non-entailment"}
                predictions = [label_mapping[line] for line in predictions]

            original = open(d_file, "r").readlines()
            original_data = []
            for line in original:
                line = json.loads(line)
                original_data.append(line['gold'])
            
            if len(predictions) != len(original_data):
                original_data = original_data[:len(predictions)]
            
            a = np.where(np.array(predictions) == np.array(original_data))[0]
            acc = len(a) / len(original_data) * 100
            # print(f"Prediction acc on {task_name}-{os.path.basename(d_file)}: {acc:.2f}")
            
            total_results.append([acc, len(predictions)])
            task_acc[task] = acc

        total_acc = sum([tup[0] * tup[1] for tup in total_results])
        total_len = sum([tup[1] for tup in total_results])
        print(f"Prediction acc on big-bench: {total_acc/total_len:.2f}")
        for task in task_acc:
            print(f"    ㄴ Prediction acc on {task}: {task_acc[task]:.2f}")

    else:
        predictions = open(os.path.join(pred_dir, f"{task_name}_predictions.txt"), "r").readlines()
        predictions = [line.strip().split('\t')[1] for line in predictions[1:]]
        if task_name in ['hans', 'wnli', 'qnli', 'nq-nli']:
            label_mapping = {"entailment": "entailment",
                             "neutral": "non-entailment",
                             "contradiction": "non-entailment"}
            predictions = [label_mapping[line] for line in predictions]
        
        original = open(data_file, "r").readlines()
        original_data = []
        for line in original:
            line = json.loads(line)
            original_data.append(line['gold'])

        a = np.where(np.array(predictions) == np.array(original_data))[0]
        acc = len(a) / len(original_data) * 100
        print(f"Prediction acc on {task_name}-{os.path.basename(data_file)}: {acc:.2f}")
print("###################################################\n")
