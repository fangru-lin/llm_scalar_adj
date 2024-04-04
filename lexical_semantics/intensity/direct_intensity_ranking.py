import pickle
from operator import itemgetter
from collections import defaultdict
import argparse
import os
from utils.utils import read_scales, load_vectors, get_scale_words
import numpy as np
from scipy.spatial.distance import cosine


def extract_gold_mildest_extreme(ranking):
    '''
    Extract the mildest and extreme words from the gold ranking
        Parameters:
            ranking (dict): gold ranking
        Returns:
            ordered_words (list): all words in the ranking
    '''
    ordered_words = []
    for i, w in enumerate(ranking):
        ordered_words.extend(w.split(" || "))
        if i == 0:
            mildest_words = w.split(" || ")
        if i == len(ranking) - 1:  # last element
            extreme_words = w.split(" || ")
    return ordered_words, mildest_words, extreme_words


def calculate_diff_vector(data,
                          scalar_dataset,
                          datanames,
                          dataname,
                          adjs_by_dataset,
                          avoid_overlap=True):
    '''
    Calculate the difference vector between the extreme and mildest words
        Parameters:
            data (dict): sentence data
            scalar_dataset (dict): gold ranking
            datanames (list): list of dataset names
            dataname (str): current dataset name
            adjs_by_dataset (dict): adjectives by dataset
            avoid_overlap (bool): whether to avoid overlapping adjectives
        Returns:
            final_diff_vector_by_layer (dict): difference vector by layer
    '''

    relevant_dds = [dataname + "-" + d for d in datanames if d != dataname]

    diff_vectors_by_layer = dict()
    pairs_by_layer = dict()
    for dd in relevant_dds:
        diff_vectors_by_layer[dd] = dict()
        pairs_by_layer[dd] = dict()
        for layer in range(1, 13):
            diff_vectors_by_layer[dd][layer] = []
            pairs_by_layer[dd][layer] = []
    missing = 0

    for scale in scalar_dataset:
        ordered_words, mildest_words, extreme_words = extract_gold_mildest_extreme(scalar_dataset[scale])
        if tuple(ordered_words) not in data:
            print(ordered_words)
            missing += 1
            continue
        mildest_word = mildest_words[0]
        extreme_word = extreme_words[0]
        for dd in relevant_dds:
            if avoid_overlap:
                if mildest_word in adjs_by_dataset[dd.split("-")[1]] or extreme_word in adjs_by_dataset[dd.split("-")[1]]:
                    continue
            diff_vectors_one_scale = dict()
            pairs_one_scale = dict()
            for layer in range(1, 13):
                diff_vectors_one_scale[layer] = []
                pairs_one_scale[layer] = []

            for instance in data[tuple(ordered_words)][:10]:
                for layer in range(1,13):

                    mild_rep = instance['representations'][mildest_word][layer].numpy()
                    extreme_rep = instance['representations'][extreme_word][layer].numpy()
                    diffvec_ex = extreme_rep - mild_rep
                    diff_vectors_one_scale[layer].append(diffvec_ex)

            for layer in range(1, 13):
                av_ex = np.average(diff_vectors_one_scale[layer], axis=0)
                diff_vectors_by_layer[dd][layer].append(av_ex)

    final_diff_vector_by_layer = dict()
    for dd in diff_vectors_by_layer:
        final_diff_vector_by_layer[dd] = dict()
        for layer in range(1,13):
            av_ex = np.average(diff_vectors_by_layer[dd][layer], axis=0)
            final_diff_vector_by_layer[dd][layer] = av_ex

    print('missing scales:', missing)
    return final_diff_vector_by_layer


def load_rankings(data_dir, datanames=["demelo", "crowd", "wilkinson"]):
    '''
    Load rankings from the gold ranking files
        Parameters:
            data_dir (str): path to the data directory
            datanames (list): list of dataset names
        Returns:
            rankings (dict): gold rankings
    '''
    rankings = dict()
    adjs_by_dataset = dict()
    for dataname in datanames:
        rankings[dataname] = read_scales(data_dir + dataname + "/gold_rankings/")
    for dataname in rankings:
        adjs_by_dataset[dataname] = set()
        for scale in rankings[dataname]:
            adjs_by_dataset[dataname].update(get_scale_words(rankings[dataname][scale]))

    return rankings, adjs_by_dataset


def assign_ranking_numbers(ordered_pred):
    '''
    Assign ranking numbers to the predictions
        Parameters:
            ordered_pred (list): ordered predictions
        Returns:
            wordscores_by_rank (dict): words and scores by rank
    '''
    ranknum = 0
    # take care of possible ties (especially for the sense baseline)
    ordered_rank_word_score = []
    for w, score in ordered_pred:
        if ordered_rank_word_score:
            if score != ordered_rank_word_score[-1][2]:
                ranknum +=1
        ordered_rank_word_score.append((ranknum, w, score))

    wordscores_by_rank = dict()
    for rank, w, score in ordered_rank_word_score:
        if rank not in wordscores_by_rank:
            wordscores_by_rank[rank] = []
        wordscores_by_rank[rank].append((w, score))

    return wordscores_by_rank


def calculate_static_diff_vector(vectors,
                                 scalar_dataset,
                                 adjs_by_dataset,
                                 datanames,
                                 dataname,
                                 avoid_overlap=True):
    '''
    Calculate the difference vector between the extreme and mildest words with static embeddings
        Parameters:
            vectors (dict): word embeddings
            scalar_dataset (dict): gold ranking
            adjs_by_dataset (dict): adjectives by dataset
            datanames (list): list of dataset names
            dataname (str): current dataset name
            avoid_overlap (bool): whether to avoid overlapping adjectives
        Returns:
            final_diff_vector (dict): difference vector
    '''
    ### dataset used to build diffvec - dataset on which we will make predictions
    relevant_dds = [dataname + "-" + d for d in datanames if d != dataname]

    diffvectors_extreme_by_scale = dict()
    pairs_by_scale = dict()

    for dd in relevant_dds:
        diffvectors_extreme_by_scale[dd] = []
        pairs_by_scale[dd] = []

    for scale in scalar_dataset:
        ordered_words, mildest_words, extreme_words = extract_gold_mildest_extreme(scalar_dataset[scale])
        mildest_word = mildest_words[0]
        extreme_word = extreme_words[0]

        mild_rep = vectors[mildest_word]
        extreme_rep = vectors[extreme_word]
        diffvec_ex = extreme_rep - mild_rep

        for dd in relevant_dds:
            if avoid_overlap:
                if mildest_word in adjs_by_dataset[dd.split("-")[1]] or extreme_word in adjs_by_dataset[dd.split("-")[1]]:
                    continue
            diffvectors_extreme_by_scale[dd].append(diffvec_ex)

    final_diff_vector = dict()
    for dd in diffvectors_extreme_by_scale:
        final_diff_vector[dd] = np.average(diffvectors_extreme_by_scale[dd], axis=0)

    return final_diff_vector


def make_staticdiffvec_prediction(scale_words, vectors, diffvec):
    '''
    Make predictions using static embeddings
        Parameters:
            scale_words (list): scale words
            vectors (dict): word embeddings
            diffvec (np.array): difference vector
        Returns:
            static_distances_to_diff (list): distances to the difference vector
    '''
    static_distances_to_diff = ((w, cosine(vectors[w], diffvec)) for w in scale_words)
    return static_distances_to_diff


def make_diffvec_prediction(scale_words, sentence_data,  diffvecs, layer):
    '''
    Make predictions using contextualized embeddings
        Parameters:
            scale_words (list): scale words
            sentence_data (dict): sentence data
            diffvecs (dict): difference vector
            layer (int): layer
        Returns:
            distances_to_diff (list): distances to the difference vector
    '''
    vectors_per_word = defaultdict(list)

    if tuple(scale_words) not in sentence_data:
        return False

    for instance in sentence_data[tuple(scale_words)][:10]:
        for w in instance['representations']:
            vectors_per_word[w].append(instance['representations'][w][layer].numpy())

    avvector_per_word = {w: np.average(vectors_per_word[w], axis=0) for w in vectors_per_word}
    avvector_per_word2 = avvector_per_word

    distances_to_diff = ((w, cosine(avvector_per_word2[w], diffvecs[layer])) for w in avvector_per_word)

    return distances_to_diff


def save_predicted_rankings(predicted_ranking_dict, args):
    '''
    Save the predicted rankings
        Parameters:
            predicted_ranking_dict (dict): predicted rankings
            args (argparse): arguments
        Returns:
            None
    '''
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    for dataname in predicted_ranking_dict:
        if not os.path.isdir(os.path.join(args.output_dir, dataname)):
            os.mkdir(os.path.join(args.output_dir, dataname))
        for scale in predicted_ranking_dict[dataname]:
            for method in predicted_ranking_dict[dataname][scale]:
                if not os.path.isdir(os.path.join(args.output_dir, dataname, method)):
                    os.mkdir(os.path.join(args.output_dir, dataname, method))

                newfilepath =os.path.join(args.output_dir, dataname, method, scale)
                with open(newfilepath, 'w') as out:
                    wordscores_by_rank = assign_ranking_numbers(predicted_ranking_dict[dataname][scale][method])
                    for rank in wordscores_by_rank:
                        ws = " || ".join([w for w, score in wordscores_by_rank[rank]])
                        scs = " || ".join([str(score) for w, score in wordscores_by_rank[rank]])
                        line = str(rank) + "\t" + ws + "\t" + scs + "\n"
                        out.write(line)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default="predictions/", type=str, required=False,
                        help="The output directory where predictions will be written")
    parser.add_argument("--num_sentences", default=10, type=int, required=False,
                        help="Number of sentences to be used to build diffvec representations")
    parser.add_argument("--embedding_path", default=None, type=str, required=True,
                        help="Path to word embeddings")
    parser.add_argument("--static", default=False, type=bool, required=False,
                        help="Whether use static embeddings")

    datanames = ['demelo', 'crowd', 'wilkinson']

    args = parser.parse_args()

    #### LOAD RANKINGS
    rankings, adjs_by_dataset = load_rankings("./data/", datanames=datanames)

    if not args.static:
        sentence_data = pickle.load(open(args.embedding_path, "rb"))
    else:
        vectors = load_vectors('./crawl-300d-2M.vec')

    diffvecs_by_dataname = dict()
    for dataname in datanames:
        if not args.static:
            diffvecs_by_dataname.update(calculate_diff_vector(sentence_data,
                                                              rankings[dataname],
                                                              datanames,
                                                              dataname,
                                                              adjs_by_dataset))
        else:
            diffvecs_by_dataname.update(calculate_static_diff_vector(vectors,
                                                                     rankings[dataname],
                                                                     adjs_by_dataset,
                                                                     datanames,
                                                                     dataname))

    # PREDICTION LOOP
    predictions = dict()
    for dataname in rankings:
        print(dataname)
        predictions[dataname] = dict()
        for scale in rankings[dataname]:
            predictions[dataname][scale] = dict()
            scale_words = get_scale_words(rankings[dataname][scale])

            for dataname_reference in datanames:
                if dataname_reference != dataname:
                    if args.static:
                        submethod = dataname_reference
                        pred = make_staticdiffvec_prediction(scale_words,
                                                             vectors,
                                                             diffvecs_by_dataname[dataname_reference + "-" + dataname])
                        ordered_pred = sorted(pred, key=itemgetter(1), reverse=True)  # the higher the distance, the milder
                        predictions[dataname][scale][submethod] = ordered_pred
                    else:
                        for layer in range(1, 13):
                            submethod = str(layer) + '-' + dataname_reference
                            pred = make_diffvec_prediction(scale_words,
                                                           sentence_data,
                                                           diffvecs_by_dataname[dataname_reference + "-" + dataname],
                                                           layer)
                            ordered_pred = sorted(pred, key=itemgetter(1), reverse=True)  # the higher the distance, the milder
                            predictions[dataname][scale][submethod] = ordered_pred

    save_predicted_rankings(predictions, args)


if __name__ == "__main__":
    main()
