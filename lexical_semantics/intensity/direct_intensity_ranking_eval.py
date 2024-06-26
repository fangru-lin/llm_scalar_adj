'''
Evaluation script for predictions obtained using predict.py
Adapted from the script at https://github.com/acocos/scalar-adj/blob/master/globalorder/src/eval.py
'''

import os, sys
import itertools
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import pdb
from copy import deepcopy
import os
import argparse
import warnings

warnings.filterwarnings("error")


def read_rankings(dirname):
    '''
    ::param: dirname : directory holding results files
    ::returns: dict  : {word: score}
    '''
    rankingfiles = os.listdir(dirname)
    results = {}
    mildextreme = dict()
    for rf in rankingfiles:
        if '.DS_Store' in rf:
            continue
        results[rf] = {}
        mildextreme[rf] = dict()
        with open(os.path.join(dirname, rf), 'r') as fin:
            ordered_words = []

            for line in fin:
                score, words = line.strip().split('\t')[:2]
                ordered_words.append(words)
                words = words.split(' || ')
                score = float(score)

                for word in words:
                    results[rf][word] = score  # this is the ranking position

            try:
                mildest_word = ordered_words[0].split(" || ")[0]
            except IndexError:
                pdb.set_trace()
            extreme_word = ordered_words[-1].split(" || ")[0]
            mildextreme[rf]['mild'] = mildest_word
            mildextreme[rf]['extreme'] = extreme_word

    return results, mildextreme


def compare_ranks(ranks):
    words = ranks.keys()
    pairs = [(i ,j) for i ,j in itertools.product(words, words) if i< j]
    compared = {}
    for p in pairs:
        w1, w2 = p
        if ranks[w1] < ranks[w2]:
            cls = "<"
        elif ranks[w1] > ranks[w2]:
            cls = ">"
        else:
            cls = "="
        compared[p] = cls
    return compared


def pairwise_accuracy(pred_rankings, gold_rankings, plustwo=False, omit_extreme_or_mild=None, gold_mildextreme=None):
    # plustwo means we only consider scales with +2 words (for BERTSIM, section 5.1 in the paper)
    results_by_pair = []
    g = []
    p = []
    for rankfile, predranks in pred_rankings.items():
        if plustwo:
            if len(predranks) <= 2:
                continue
        goldranks = gold_rankings.get(rankfile, None)
        if goldranks is None:
            sys.stderr.write('Acc - Could not find gold rankings for cluster %s\n' % rankfile)
            continue

        goldranks = deepcopy(goldranks)
        predranks = deepcopy(predranks)
        if omit_extreme_or_mild:
            word_to_remove = gold_mildextreme[rankfile][omit_extreme_or_mild]
            del goldranks[word_to_remove]
            del predranks[word_to_remove]

        gold_compare = compare_ranks(goldranks)
        pred_compare = compare_ranks(predranks)
        gold_compare = deepcopy(gold_compare)
        pred_compare = deepcopy(pred_compare)

        if not gold_compare:
            continue

        if set(gold_compare.keys()) != set(pred_compare.keys()):
            sys.stderr.write('Acc - ERROR: Key mismatch for cluster %s\n' % rankfile)
            continue
        pairs = sorted(list(gold_compare.keys()))
        for pr in pairs:
            g.append(gold_compare[pr])
            p.append(pred_compare[pr])
            if gold_compare[pr] == pred_compare[pr]:
                results_by_pair.append((pr, gold_compare[pr], pred_compare[pr], 1))
            else:
                results_by_pair.append((pr, gold_compare[pr], pred_compare[pr], 0))


    correct = [1. if gg == pp else 0. for gg, pp in zip(g, p)]

    acc = sum(correct) / len(correct)

    return acc, results_by_pair


def kendalls_tau(pred_rankings, gold_rankings, plustwo=False, omit_extreme_or_mild=None,gold_mildextreme=None):
    # plustwo means we only consider scales with +2 words (for BERTSIM, section 5.1 in the paper)
    results_by_scale = dict()
    taus = []
    ns = []
    for rankfile, predranks in pred_rankings.items():
        if plustwo:
            if len(predranks) <= 2:
                continue
        n_c = 0.
        n_d = 0.
        n = 0.
        ties_g = 0.
        ties_p = 0.
        goldranks = gold_rankings.get(rankfile, None)
        if goldranks is None:
            sys.stderr.write('Kendall - Could not find gold rankings for cluster %s\n' % rankfile)
            continue
        goldranks = deepcopy(goldranks)
        predranks = deepcopy(predranks)
        if omit_extreme_or_mild:
            word_to_remove = gold_mildextreme[rankfile][omit_extreme_or_mild]
            del goldranks[word_to_remove]
            del predranks[word_to_remove]

        words = sorted(goldranks.keys())
        for w_i in words:
            for w_j in words:
                if w_j >= w_i:
                    continue
                n += 1
                # check ties
                tied = False
                if goldranks[w_i] == goldranks[w_j]:
                    ties_g += 1
                    tied = True
                if predranks[w_i] == predranks[w_j]:
                    ties_p += 1
                    tied = True
                if tied:
                    continue
                # concordant/discordant
                dir_g = np.sign(goldranks[w_j] - goldranks[w_i])
                dir_p = np.sign(predranks[w_j] - predranks[w_i])
                if dir_g == dir_p:
                    n_c += 1
                else:
                    n_d += 1
        try:
            tau = (n_c - n_d) / np.sqrt((n - ties_g) * (n - ties_p))
        except RuntimeWarning:
            tau = 0.

        results_by_scale[rankfile] = (words, tau)
        taus.append(tau)
        ns.append(n)

    taus = [t if not np.isnan(t) else 0. for t in taus]
    tau_avg = np.average(taus, weights=ns)

    return tau_avg, results_by_scale


def spearmans_rho_avg(pred_rankings, gold_rankings, plustwo=True, omit_extreme_or_mild=None, gold_mildextreme=None):
    # plustwo means we only take into account scales with +2 words
    results_by_scale = dict()
    rhos = []
    ns = []
    for rankfile, predranks in pred_rankings.items():
        if plustwo:
            if len(predranks) <= 2:
                continue
        goldranks = gold_rankings.get(rankfile, None)
        if goldranks is None:
            sys.stderr.write('Spearman -Could not find gold rankings for cluster %s\n' % rankfile)
            continue

        if set(goldranks.keys()) != set(predranks.keys()):
            pdb.set_trace()
            sys.stderr.write('Spearman -ERROR: Key mismatch for cluster %s\n' % rankfile)
            continue

        goldranks = deepcopy(goldranks)
        predranks = deepcopy(predranks)

        if omit_extreme_or_mild:
            word_to_remove = gold_mildextreme[rankfile][omit_extreme_or_mild]
            del goldranks[word_to_remove]
            del predranks[word_to_remove]

        ns.append(len(goldranks))
        words = sorted(goldranks.keys())

        predscores = [predranks[w] for w in words]
        goldscores = [goldranks[w] for w in words]

        try:
            r, p = spearmanr(predscores, goldscores)
        except RuntimeWarning:
            r = 0.
        if np.isnan(r):
            r = 0.
        rhos.append(r)
        results_by_scale[rankfile] = (words, r)

    rho = np.average(rhos, weights=ns)

    return rho, results_by_scale



def save_results(results_dict, args):
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for dataname in results_dict:
        if not os.path.isdir(os.path.join(output_dir, dataname)):
            os.mkdir(os.path.join(output_dir, dataname))
        for sentence_type in results_dict[dataname]:

            if not os.path.isdir(os.path.join(output_dir, dataname, sentence_type)):
                os.mkdir(os.path.join(output_dir, dataname, sentence_type))

            newfilename = os.path.join(output_dir, dataname, sentence_type) + "/results.csv"

            df = pd.DataFrame.from_dict(results_dict[dataname][sentence_type], orient='index')

            df.to_csv(newfilename, sep="\t")


def calculate_coverage(pred_rankings, gold_rankings):
    total_len = len(gold_rankings)
    coverage = 0
    for rf, _ in gold_rankings.items():
        if rf in pred_rankings:
            coverage += 1
    coverage = coverage / total_len

    return coverage


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data/", type=str,
                        help="Folder containing the scalar adjective datasets.")
    parser.add_argument("--pred_dir", default="predictions/", type=str,
                        help="Folder containing the predictions for the scalar adjective datasets")
    parser.add_argument("--output_dir", default="results/", type=str,
                        help="Folder to store the evaluation results.")

    args = parser.parse_args()

    datanames = ['demelo', 'crowd', 'wilkinson']

    pred_dir = args.pred_dir

    results = dict()
    results_by_scale = dict()

    ####### EVALUATION LOOP
    for dataname in datanames:
        print(dataname)
        gold_rankings, gold_mildextreme = read_rankings(
            args.data_dir + dataname + "/gold_rankings/")  # load gold rankings
        results[dataname] = dict()
        results_by_scale[dataname] = dict()
        diri = os.path.join(pred_dir, dataname)
        best_perf = {'pacc': dict(), 'tau': dict(), 'rho': dict()}

        for method in os.listdir(diri):
            if '.DS_Store' in method:
                continue
            pred_rankings, _ = read_rankings(os.path.join(diri, method))
            results[dataname][method] = dict()
            results_by_scale[dataname][method] = dict()
            results[dataname][method]['pacc'], results_by_scale[dataname][method]['pacc'] = pairwise_accuracy(pred_rankings,gold_rankings)
            results[dataname][method]["tau"], results_by_scale[dataname][method]['tau'] = kendalls_tau(pred_rankings, gold_rankings)
            results[dataname][method]["rho"], results_by_scale[dataname][method]['rho'] = spearmans_rho_avg(pred_rankings, gold_rankings)
            dvec = method.split('-')[-1]
            if dvec in best_perf['pacc']:
                for key in best_perf.keys():
                    best_perf[key][dvec] = max(results[dataname][method][key], best_perf[key][dvec])
            else:
                for key in best_perf.keys():
                    best_perf[key][dvec] = results[dataname][method][key]

            ## Coverage
            coverage = calculate_coverage(pred_rankings, gold_rankings)
            results[dataname][method]["coverage"] = coverage

        print(best_perf)

    save_results(results, args)
