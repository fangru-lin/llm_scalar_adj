from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import sklearn
import argparse
from os import listdir


def read_results(result_dir):
    '''
    Read a bunch of model outputs in TSV SyntaxGym format.
        Parameters:
            result_dir (str): directory containing model outputs
        Return:
            df (pd.DataFrame): dataframe containing all model outputs
    '''
    files = sorted([f for f in listdir(result_dir) if f.endswith("gpt2.tsv")])
    dfs = []
    sentence_dfs = []
    for f in files:
        try:
            _df = pd.read_csv(result_dir + "/" + f, sep="\t")
            weak, strong, model = f[:-4].split("_")
            scale_id = weak + "/" + strong
            _df["scale_id"] = scale_id
            _df["model"] = model
            dfs.append(_df)

            g = _df.fillna("").groupby("item_number").agg({'content': " ".join}).reset_index()
            g["scale_id"] = scale_id
            g["content"] = g.content.str.replace(" .", ".", regex=False).replace("\s+,", ",", regex=True)
            sentence_dfs.append(g)
        except:
            print(f"Skipping {f}")
    df = pd.concat(dfs)
    sentence_df = pd.concat(sentence_dfs)
    return df, sentence_df


def get_strong_surprisal_df(surprisal_df, region_numbers=[5]):
    '''
    Get dataframe with surprisal summed over specified regions.
        Parameters:
            surprisal_df (pd.DataFrame): dataframe containing surprisal values
            region_numbers (list): list of region numbers to consider
        Return:
            strong_surprisal_df (pd.DataFrame): dataframe with surprisal summed over specified regions
    '''
    print("Only considering surprisal at the following regions:", region_numbers)

    # Get subset of surprisals that correspond to specified region numbers.
    strong_surprisal_df = surprisal_df[surprisal_df.region_number.isin(region_numbers)].fillna("")

    # Sum surprisal and concatenate string content over regions of interest.
    strong_surprisal_df = strong_surprisal_df.groupby(
        ["scale_id", "item_number"]
    ).agg(
        {"content": " ".join, "metric_value": np.sum}
    ).reset_index()

    # Deal with edge case (removing extra space before EOS period).
    strong_surprisal_df["content"] = strong_surprisal_df.content.str.replace(" .", ".", regex=False)

    # Add unique ID for each item.
    strong_surprisal_df["unique_id"] = strong_surprisal_df.scale_id + "_" + strong_surprisal_df.item_number.astype(str)

    return strong_surprisal_df


def process_model_outputs(result_dir, stimuli_file, model="gpt2", template=None):
    '''
    Process model outputs and add surprisal values to stimuli dataframe.
        Parameters:
            result_dir (str): directory containing model outputs
            stimuli_file (str): file containing stimuli
            model (str): name of model
            template (int): template number
        Return:
            stims (pd.DataFrame): dataframe containing stimuli with surprisal values
    '''
    # Read model outputs and only consider surprisal at target region.
    df, sentence_df = read_results(result_dir)
    strong_surp = get_strong_surprisal_df(df).reset_index()

    # Read original data file with stimuli and human empirical results.
    print(f"Reading stimuli from {stimuli_file}")
    stims = pd.read_csv(stimuli_file)
    if "scale_id" not in list(stims):
        try:
            stims["scale_id"] = stims.Weaker + "/" + stims.Stronger
        except:
            stims["scale_id"] = stims.weak_adj + "/" + stims.strong_adj

    # Add surprisal results.
    print("Adding strong scalemate surprisal results")
    strong_surp["strong"] = strong_surp.scale_id.str.split('/').str[1]
    strong_surp["is_strong_scalemate"] = strong_surp.apply(
        lambda r: r.content==r.strong+"." or r.content.split()[0]==r.strong or f" {r.strong}" in r.content, 
        axis=1
    )
    s = strong_surp[strong_surp.is_strong_scalemate].set_index("scale_id")
    if template is None:
        surprisal_col_name = f"surprisal_{model}"
    else:
        surprisal_col_name = f"surprisal_{model}_template{template}"
    stims[surprisal_col_name] = stims.apply(
        lambda row: s.loc[row.scale_id].metric_value if row.scale_id in s.index else None, 
        axis=1
    )

    # Standardize dependent variable name across datasets.
    stims = stims.rename(columns={
        "SI percent (Exp 1)": "si_rate",
        "SI_rate": "si_rate",
        "si_nonneutral": "si_rate"
    })

    return stims, strong_surp, sentence_df


def lr_two_feature(train_sets, test_set, data_dic):
    '''
    Compute F1 score for logistic regression with two features.
        Parameters:
            train_sets (list or str): list of training sets or single training set
            test_set (str): test set
            data_dic (dict): dictionary containing dataframes
        Return:
            f1 (float): F1 score
    '''
    if type(train_sets) is list:
        train1, train2 = train_sets
        X = np.array(pd.concat([data_dic[train1][['surprisal_gpt2', 'weighted_avg_surprisal']],
                               data_dic[train2][['surprisal_gpt2', 'weighted_avg_surprisal']]]))
        y = np.array(pd.concat([data_dic[train1]['si_binary'], data_dic[train2]['si_binary']]))
    else:
        X = np.array(data_dic[train_sets][['surprisal_gpt2', 'weighted_avg_surprisal']])
        y = np.array(data_dic[train_sets]['si_binary'])
        print(train_sets, len(data_dic[train_sets]), X.shape[0], y.shape[0])
    clf = LogisticRegression(random_state=0, class_weight='balanced').fit(X, np.array(y))
    predict = clf.predict(np.array(data_dic[test_set][['surprisal_gpt2', 'weighted_avg_surprisal']]))
    f1 = sklearn.metrics.f1_score(data_dic[test_set]['si_binary'], predict, average='macro')

    return f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    rx = process_model_outputs(f"{args.data_dir}/model_results/rx22", 
                               f"{args.data_dir}/human_data/rx22.csv")[0]
    gz = process_model_outputs(f"{args.data_dir}/model_results/g18",
                               f"{args.data_dir}/human_data/g18.csv")[0]
    pvt = process_model_outputs(f"{args.data_dir}/model_results/pvt21",
                                f"{args.data_dir}/human_data/pvt21.csv")[0]
    reduced_dic = dict()

    reduced_dic['gz'] = gz[['si_rate', 'dist', 'surprisal_gpt2', 'weighted_avg_surprisal']][gz['pos']=='adj'][gz['pos']=='adj']
    reduced_dic['pvt'] = pvt[['si_rate', 'w2v_simil', 'surprisal_gpt2', 'weighted_avg_surprisal']][pvt['pos']=='adj']
    reduced_dic['rx'] = rx[['si_rate', 'Distinctness (Exp 3)', 'surprisal_gpt2', 'weighted_avg_surprisal']][rx['pos']=='adj']

    no_na_dic = dict()
    for key, val in reduced_dic.items():
        no_na_dic[key] = val.dropna()
    for i, test_set in enumerate(['rx', 'gz', 'pvt']):
        print(f'test set is {test_set}')
        max_f1_stra = [0, []]
        for train_set in ['gz', 'pvt', 'rx'][:i]+['gz', 'pvt', 'rx'][i+1:]:
            f1 = lr_two_feature(train_set, test_set, no_na_dic)
            if f1 > max_f1_stra[0]:
                max_f1_stra[0] = f1
                max_f1_stra[1] = [[train_set]]
            elif f1==max_f1_stra[0]:
                max_f1_stra[1].append([train_set])
        f1 = lr_two_feature(['gz', 'pvt', 'rx'][:i]+['gz', 'pvt', 'rx'][i+1:], test_set, no_na_dic)
        if f1 > max_f1_stra[0]:
            max_f1_stra[0] = f1
            max_f1_stra[1] = [['all']]
        elif f1 == max_f1_stra[0]:
            max_f1_stra[1].append(['all'])
        print(max_f1_stra)
        with open(f'{args.output_dir}/lr_two_feature_{test_set}.txt', 'w') as f:
            f.write(str(max_f1_stra))
