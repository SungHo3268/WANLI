"""
Download data from https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/disambiguation_qa
and store in data/disambiguation_qa/raw
"""

import pandas as pd
import json
from tqdm.auto import tqdm

task = "disambiguation_qa"

with open(f'data/{task}/raw/task.json', 'r') as fin:
    data = fin.read()

data = json.loads(data)

nli_examples = []
examples = data['examples']
for row in tqdm(examples, desc=f"Preprocessing {task} dataset...", total=len(examples), bar_format="{l_bar}{bar:20}{r_bar}"):
    premise = row['input'].strip()
    d = row["target_scores"]

    hypothesis = []
    gold = []
    for key in d:
        if key == 'Ambiguous':
            continue
        hypothesis.append(key)
        gold.append(d[key])
    if d['Ambiguous'] == 1:
        gold = ['neutral' for _ in range(len(hypothesis))]

    gold_mapping = {0: 'contradiction', 1: 'entailment', 'neutral': 'neutral'}

    for i in range(len(hypothesis)):
        ex = {
            'premise': premise,
            'hypothesis': hypothesis[i],
            'gold': gold_mapping[gold[i]],
            "genre": "dummy",
            "pairID": 0
        }
        nli_examples.append(ex)

df = pd.DataFrame(nli_examples)
df = df.reset_index().drop('index', axis=1).reset_index().rename(columns={'index': 'id'})

df.to_json(f'data/{task}/dev.jsonl', orient='records', lines=True)
