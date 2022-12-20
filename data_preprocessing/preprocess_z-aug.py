import pandas as pd
import jsonlines

z_aug = []
label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
with jsonlines.open("data/z-aug/raw/mnli_z-aug.jsonl") as f:
    for line in f.iter():
        line['label'] = label_map[line['label']]
        z_aug.append(line)
z_aug = pd.DataFrame(z_aug)
z_aug = z_aug.rename(columns={'label': 'gold', 'type': 'genre'})
z_aug['pairId'] = 0
z_aug = z_aug.reset_index().drop('index', axis=1).reset_index().rename(columns={'index': 'id'})

z_aug.to_json('data/z-aug/train.jsonl', lines=True, orient='records')
