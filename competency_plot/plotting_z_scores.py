import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math
import random
import argparse
import os


add_jitter = True

seed = 6
np.random.seed(seed)

prob2z = {
    "0.01*(1/28k)": 4.9578,
}

label_to_marker = {"contradiction": "+",
                   "neutral": ".",
                   "entailment": "x"}

colors = {
    "neutral": "orange",
    "contradiction": "red",
    "entailment": "green",
}

label_order = ["neutral", "contradiction", "entailment"]


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

    
def annotate_points(subdata, total_counts, p_hat, new_zs, label):
    # just to explore the largest z-score points: 
    # filtered = [(t, n, p) for t, n, p, z in zip(subdata["tokens"], total_counts, p_hat, new_zs) if n > 20 and z > 25.9578]
    
    # points to annotate:
    if label == "contradiction":
        tokens_to_anno = ["nobody", "sleeping", "no", "cat", "cats"]
    elif label == "entailment":
        tokens_to_anno = ["outdoors", "person", "people", "outside", "animal"]
    elif label == "neutral":
        tokens_to_anno = ["vacation", "first", "competition"]
    else:
        assert label in ["contradiction", "entailment", "neutral"]
        
        
    pts_to_annotate = [(token, n, p) for token, n, p, in zip(subdata["tokens"], total_counts, p_hat) if token in tokens_to_anno]
    for pt in pts_to_annotate:
        if pt[0] == "competition":
            pt = (pt[0], pt[1]-10, pt[2]-0.015)
        annotation_string = "$" + pt[0] + "^" + label[0] + "$"
        plt.annotate(annotation_string, (pt[1], pt[2]+0.005))

def format_data_and_plot(data):
    max_n = 0
    for label in label_order:
        subdata = data[label]
        total_counts = subdata["total_counts"]
        p_hat = subdata["p_hat"]
        
        if add_jitter:
            p_hat_jit = 0.03 * np.random.rand(len(p_hat)) - 0.015
            p_hat = p_hat_jit + p_hat
            
            total_counts_jit = np.random.rand(len(p_hat)) - 0.5
            total_counts = total_counts_jit + total_counts
        
        # zs = subdata["z"]

        new_zs = [(p - 1/3) / (math.sqrt((1/3)*(1-1/3)/n)) for n, p in zip(total_counts, p_hat)]
        threshold = 3.9578     # 4.9578
        filtered = [(n, p) for n, p, z in zip(total_counts, p_hat, new_zs) if n > 20 and z > threshold]
        max_n = max(max(total_counts), max_n)
        
        random.shuffle(filtered)
        
        if label == "neutral":
            marker_size = 2
        else:
            marker_size = 3
        plt.scatter(*zip(*filtered), marker=label_to_marker[label], color=colors[label], alpha=.5, label=label[0:10], s=marker_size)

        filtered_small = [(n, p) for n, p, z in zip(total_counts, p_hat, new_zs) if n > 20 and z < threshold]
        plt.scatter(*zip(*filtered_small), marker='.', color="grey", alpha=.1, s=3)

        annotate_points(subdata, total_counts, p_hat, new_zs, label)
    return max_n, threshold


def label_plot_and_save(max_n, threshold, dataset_name):

    ns = np.logspace(np.log(20)/np.log(10), np.log(max_n) / np.log(10), 1000)
    
    for p, z in prob2z.items():
        p0 = 1/3
        qs = p0 + z * np.sqrt(p0 * (1 - p0)) / np.sqrt(ns)
        plt.plot(ns, qs, label=r"$\alpha=0.01/28k$")

    plt.title(fR"Artifact statistics in {dataset_name.upper()}")
    plt.ylabel(R"$\hat p(y \mid x_i)$")
    plt.xlabel(R"$n$")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout(pad=0)
    plot_location = "competency_plot/figs/"
    os.makedirs(plot_location, exist_ok=True)
    plot_name = f"{dataset_name}_{threshold}_plot.png"
    save_name = os.path.join(plot_location, plot_name)
    
    print("Saving in {}".format(save_name))
    plt.savefig(save_name)


def main(path_to_data, dataset_name):
    with open(path_to_data, "rb") as fh:
        data = pickle.load(fh)
    
    max_n, threshold = format_data_and_plot(data)
    label_plot_and_save(max_n, threshold, dataset_name)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--host', type=str, default='dummy')
    parser.add_argument('--port', type=int, default=56789)
    parser.add_argument('--dataset_name', type=str, default="mnli", help="mnli  |  wanli")
    parser.add_argument("--data_dir", type=str, default="competency_plot/scores/", help="path to the file containing the data to plot.")
    args = parser.parse_args()

    path_to_data = os.path.join(args.data_dir, f"{args.dataset_name}_all_data.data")

    main(path_to_data, args.dataset_name)
