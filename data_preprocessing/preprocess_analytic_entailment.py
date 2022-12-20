"""
Download data from https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/analytic_entailment
and store in data/analytic_entailment/raw
"""

import pandas as pd
import json
from tqdm.auto import tqdm

task = "analytic_entailment"

with open(f'data/{task}/raw/task.json', 'r') as fin:
    data = fin.read()

data = json.loads(data)

nli_examples = []
examples = data['examples']
for row in tqdm(examples, desc=f"Preprocessing {task} dataset...", total=len(examples), bar_format="{l_bar}{bar:20}{r_bar}"):
    _input = row['input'].strip()
    separator = ['So ', 'So, ', 'Therefore ']
    premise, hypothesis = None, None
    for sep in separator:
        if sep in _input:
            premise, hypothesis = _input.split(sep)
            break
        else:
            continue
    gold = 'entailment' if row['target_scores']['entailment'] == 1 else 'non-entailment'
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
