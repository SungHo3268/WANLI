import operator
import math
from collections import Counter
import pickle
import argparse
import os
import json




counts = {}
label_counts = {}


punc_to_remove = '!.()[]{};:\,?\'\"'

type_not_token = False
if type_not_token:
    print("counting tokens in each example, so if a token appears more than once the counter increments more than once")
else:
    print("counting types in each example, so if a token appears more than once the counter increments only once")


def get_vocab_and_z_scores(lines, label_index, input_indices, dataset):
    vocab = {}

    counter = 0
    for line in lines:
        counter += 1
        
        inputs = [line[input_indices[0]].strip(), line[input_indices[1]].strip()]
        
        # NLI examples each have two sentences as input.
        first_input = add_to_vocab(vocab, inputs[0])
        second_input = add_to_vocab(vocab, inputs[1])

        
        label = line[label_index].strip()
        if label not in counts:
            counts[label] = Counter()


        if type_not_token:
            all_input = first_input.union(second_input)
        else:
            all_input = first_input + second_input


        for token in all_input:
            counts[label][token] += 1
        if counter % 10000 == 0:
            print(counter)
        
    compute_z_scores(counts, dataset)

    
def add_to_vocab(vocab, text):
    if type_not_token:
        to_return = set()
    else:
        to_return = []
    for token in text.split():
        token = token.lower()
        for punc in punc_to_remove:
            token = token.replace(punc, "")
        if token not in vocab:
            vocab[token] = 0
        
        if type_not_token:
            to_return.add(token)
        else:
            to_return.append(token)
    return to_return


def compute_z_scores(counts, dataset):
    total_counts = Counter()
    for label in counts:
        total_counts = total_counts + counts[label]


    all_z_scores = {}
    for label in counts:

        all_z_scores[label] = {}
        
        success_prob = 1.0/len(counts)
        for cur_token in total_counts:
            cur_p_hat = counts[label][cur_token] * 1.0 / total_counts[cur_token]

            cur_z_score_numerator = (cur_p_hat - success_prob)
            cur_z_score_denominator = math.sqrt( (success_prob * (1-success_prob)) / total_counts[cur_token])
            cur_z_score = cur_z_score_numerator / cur_z_score_denominator

            all_z_scores[label][cur_token] = cur_z_score
    print_z_scores(all_z_scores, total_counts, counts, dataset)

    # to save data
    save_data(all_z_scores, total_counts, counts, dataset)
    

def save_data(all_z_scores, total_counts, counts, dataset):
    to_pickle = {}
    for label in counts:
        tokens_pickle = []
        total_counts_pickle = []
        z_pickle = []
        label_count_pickle = []
        p_hat_pickle = []
        for token in counts[label]:
            tokens_pickle.append(token)
            total_counts_pickle.append(total_counts[token])

            if all_z_scores[label][token] is None:
                import pdb; pdb.set_trace()
                
            z_pickle.append(all_z_scores[label][token])
            if counts[label][token] is None:
                import pdb; pdb.set_trace()
            label_count_pickle.append(counts[label][token])
            p_hat_pickle.append(counts[label][token] * 1.0 / total_counts[token])
        to_pickle[label] = {"tokens": tokens_pickle,
                            "total_counts": total_counts_pickle,
                            "z": z_pickle,
                            "label_count": label_count_pickle,
                            "p_hat": p_hat_pickle}

    save_dir = "competency_plot/scores"
    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.join(save_dir, dataset + "_all_data.data")
    print("saving to {}".format(save_name))

    pickle.dump(to_pickle, open(save_name, "wb"))
    
    

def print_z_scores(all_z_scores, total_counts, counts, dataset):
    print("")
    print(dataset)

    tmp_counter = 0
    for item in total_counts:
        if total_counts[item] > 20:
            tmp_counter = tmp_counter + 1
    
    for label in counts:
        print("")
        print("label: " + label)
        sorted_by_z = sorted(all_z_scores[label].items(), key=operator.itemgetter(1))

        print("largest z-scores for label {}".format(label))
        print("")
        print("token, z-score, count with {}, total count with any label, p_hat".format(label))
        for i in range(10): print(sorted_by_z[i], counts[label][sorted_by_z[i][0]], total_counts[sorted_by_z[i][0]],
                                  counts[label][sorted_by_z[i][0]] * 1.0 / total_counts[sorted_by_z[i][0]])
        print("")
        
        print("smallest z-scores for label {}".format(label))
        print("")
        print("token, z-score, count with {}, total count with any label, p_hat".format(label))
        for i in range(1,11): print(sorted_by_z[-i], counts[label][sorted_by_z[-i][0]], total_counts[sorted_by_z[-i][0]],
                                    counts[label][sorted_by_z[-i][0]] * 1.0 / total_counts[sorted_by_z[-i][0]])
        print("")

        print("the tokens with the largest overall count, with z-score for {}".format(label))
        sorted_by_n = sorted(counts[label].items(), key=operator.itemgetter(1))
        print("")
        print("token, count with {}, z-score, total count with any label, p_hat".format(label))
        for i in range(1,11): print(sorted_by_n[-i], all_z_scores[label][sorted_by_n[-i][0]], total_counts[sorted_by_n[-i][0]],
                                    counts[label][sorted_by_n[-i][0]] * 1.0 /  total_counts[sorted_by_n[-i][0]])
        print("")


# these assume the GLUE dataset's tsv formatting.
def data_path_and_indices(path, dataset):
    file_to_load = os.path.join(path, dataset, 'train.jsonl')

    label_index = 'gold'
    input_indices = ['premise', 'hypothesis']

    return file_to_load, label_index, input_indices
    
        
def main(args):

    file_to_load, label_index, input_indices = data_path_and_indices(args.path_to_data, args.dataset_name)
    
    lines = []
    for line in open(file_to_load, 'r').readlines():
        lines.append(json.loads(line))
    
    get_vocab_and_z_scores(lines, label_index, input_indices, args.dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--host', type=str, default='dummy')
    parser.add_argument('--port', type=int, default=56789)
    parser.add_argument("--path_to_data", type=str, default="data/", help="the filepath to the directory containing the data.")
    parser.add_argument("--dataset_name", type=str, default="wanli", help="the supported datasets are: mnli, wanli")
    
    args = parser.parse_args()
    if args.dataset_name not in ["mnli", "wanli"]:
        assert False

    main(args)
