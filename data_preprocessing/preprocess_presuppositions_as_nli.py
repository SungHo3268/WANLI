"""
Download data from https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/presuppositions_as_nli
and store in data/presuppositions_as_nli/raw
"""

import pandas as pd
import json
from tqdm.auto import tqdm

task = "presuppositions_as_nli"

with open(f'data/{task}/raw/task.json', 'r') as fin:
    data = fin.read()

data = json.loads(data)

nli_examples = []
examples = data['examples']
for row in tqdm(examples, desc=f"Preprocessing {task} dataset...", total=len(examples), bar_format="{l_bar}{bar:20}{r_bar}"):
    s = row['input'].split('Sentence 1: ')[1]
    premise, hypothesis = s.split('Sentence 2: ')
    premise = premise.strip()
    hypothesis = hypothesis.split("The answer is: ")[0].strip()
    gold = None
    if row['target_scores']['entailment'] == 1:
        gold = 'entailment'
    elif row['target_scores']['neutral'] == 1:
        gold = 'neutral'
    elif row['target_scores']['contradiction'] == 1:
        gold = 'contradiction'
    ex = {
        'premise': premise,
        'hypothesis': hypothesis,
        'gold': gold,
        "genre": "dummy",
        "pairID": 0
    }
    nli_examples.append(ex)

df = pd.DataFrame(nli_examples)
df = df.reset_index().drop('index', axis=1).reset_index().rename(columns={'index': 'id'})

df.to_json(f'data/{task}/dev.jsonl', orient='records', lines=True)
