from sentence_transformers import SentenceTransformer, util
import argparse
import os
import json
from tqdm.auto import tqdm
import torch
import numpy as np


parser = argparse.ArgumentParser("Semantic Similarity")
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--host', type=str, default='dummy')
parser.add_argument('--port', type=int, default=56789)
parser.add_argument('--dataset_name', type=str, default="mnli", help="mnli  |  wanli")
parser.add_argument("--data_dir", type=str, default="data/scores/", help="path to the file containing the data to plot.")
args = parser.parse_args()


path_to_data = os.path.join("data", args.dataset_name, 'train.jsonl')
sentences = {'entailment': [], 'neutral': [], 'contradiction': []}
for line in tqdm(open(path_to_data, 'r').readlines(), bar_format="{l_bar}{bar:20}{r_bar}", desc=f"Loading {args.dataset_name.upper()}..."):
    instance = json.loads(line)
    label = instance['gold']
    sentences[label].append([instance['premise'],
                             instance['hypothesis']])


model = SentenceTransformer('all-MiniLM-L6-v2')

batch_size = 32
similarity = {'entailment': [], 'neutral': [], 'contradiction': []}
for relation in sentences:
    sen_list = sentences[relation]
    batch_num = len(sen_list) // batch_size if len(sen_list) % batch_size == 0 else len(sen_list) // batch_size + 1
    for i in tqdm(range(batch_num), bar_format="{l_bar}{bar:20}{r_bar}", desc=f"- Encoding '{relation}' class..."):
        batch = np.array(sen_list[i*batch_size: (i+1)*batch_size])
        pre = batch[:, 0]
        hyp = batch[:, 1]
        
        pre_embedding = model.encode(pre, convert_to_tensor=True)
        hyp_embedding = model.encode(hyp, convert_to_tensor=True)
        
        cos_sim = util.cos_sim(pre_embedding, hyp_embedding)
        cos_sim = [float(cos_sim[j, j]) for j in range(len(cos_sim))]

        similarity[relation].extend(cos_sim)
    

        
        


