import numpy as np
import argparse
from scipy.spatial.distance import cosine
from utils.utils import read_scales, read_all_adj_embeddings, generate_selected_embedding
import random
import pickle
import os
import io


def compute_scale_vec(selected_embeddings,
                      model_layer=None,
                      strategy='end',
                      embed_type='contextual',
                      static_embedding=None):
    '''
    Compute scale vectors for each scale
        Parameters:
            selected_embeddings: dict
                Adjective embeddings
            model_layer: int
                Number of layers of the model
            strategy: str
                Strategy for computing dVec
            embed_type: str
                Type of embeddings
            static_embedding: dict
                Static embeddings
        Returns:
            scale_vec: dict
    '''
    scale_vec = dict()
    for data_name, data in selected_embeddings.items():
        scale_vec[data_name] = dict()
        for scale in data.keys():
            scale_vec[data_name][scale] = dict()
            if embed_type == 'contextual':
                if strategy == 'end':
                    # Weakest and strongest adjectives on a scale
                    a_w, a_s = scale[0], scale[-1]
                    for layer in range(1, model_layer+1):
                        # Add representations of a_w and a_s in every layer
                        scale_vec[data_name][scale][layer] = data[scale][a_w][layer]+data[scale][a_s][layer]
                elif strategy == 'all':
                    for layer in range(1, model_layer+1):
                        # Add representations of all adjs in every layer
                        scale_vec[data_name][scale][layer] = data[scale]
                else:
                    raise ValueError('Invalid strategy')
            elif embed_type == 'static':
                if strategy == 'end':
                    # Weakest and strongest adjectives on a scale
                    a_w, a_s = scale[0], scale[-1]
                    mild_rep = static_embedding[a_w]
                    extreme_rep = static_embedding[a_s]
                    scale_vec[data_name][scale] = mild_rep + extreme_rep
                elif strategy == 'all':
                    scale_vec[data_name][scale] = dict()
                    for adj in scale:
                        scale_vec[data_name][scale][adj] = static_embedding[adj]
                else:
                    raise ValueError('Invalid strategy')
            else:
                raise ValueError('Invalid embed_type')

    return scale_vec


def classify_scale(selected_embeddings,
                   scale_vec,
                   model_layer=None,
                   strategy='end',
                   embed_type='contextual',
                   static_embedding=None):
    '''
    Classify adjectives to scales based on their similarity with scale vectors
        Parameters:
            selected_embeddings: dict
                Adjective embeddings
            scale_vec: dict
                Scale vectors
            model_layer: int
                Number of layers of the model
            strategy: str
                Strategy for computing dVec
            embed_type: str
                Type of embeddings
            static_embedding: dict
                Static embeddings
        Returns:
            adjs: dict
    '''
    adjs = dict()
    if embed_type == 'contextual':
        for layer in range(1, model_layer+1):
            adjs[layer] = dict()
            for data_name, data in selected_embeddings.items():
                curr_scale_vec = scale_vec[data_name]
                adjs[layer][data_name] = dict()
                for scale, scale_adjs in data.items():
                    for adj in scale:
                        cos_a_s = list()
                        for vec_name in curr_scale_vec.keys():
                            if strategy == 'end':
                                cos = cosine(curr_scale_vec[vec_name][layer], scale_adjs[adj][layer])
                            elif strategy == 'all':
                                cos = 0
                                count = 0
                                for vec_adj, vec in curr_scale_vec[vec_name][layer].items():
                                    if vec_adj == adj and vec_name == scale:
                                        continue
                                    cos += cosine(vec[layer], scale_adjs[adj][layer])
                                    count += 1
                                cos /= count
                            cos_a_s.append((cos, vec_name))
                        cos_a_s.sort()
                        for idx, item in enumerate(cos_a_s):
                            if item[-1] == scale:
                                scale_rank = idx
                                break
                        adjs[layer][data_name][(scale, adj)] = {'cos_rank': cos_a_s,
                                                                'alignment_rank': scale_rank}
    elif embed_type == 'static':
        adjs = dict()
        for data_name, data in selected_embeddings.items():
            curr_scale_vec = scale_vec[data_name]
            adjs[data_name] = dict()
            for scale, scale_adjs in data.items():
                for adj in scale:
                    cos_a_s = list()
                    for vec_name in curr_scale_vec.keys():
                        if strategy == 'end':
                            cos = cosine(curr_scale_vec[vec_name], static_embedding[adj])
                        elif strategy == 'all':
                            cos = 0
                            count = 0
                            for vec_adj, vec in curr_scale_vec[vec_name].items():
                                if vec_adj == adj and vec_name == scale:
                                    continue
                                cos += cosine(vec, static_embedding[adj])
                                count += 1
                            cos /= count
                        cos_a_s.append((cos, vec_name))
                    cos_a_s.sort()
                    for idx, item in enumerate(cos_a_s):
                        if item[-1] == scale:
                            scale_rank = idx
                            break
                    adjs[data_name][(scale, adj)] = {'cos_rank': cos_a_s,
                                                     'alignment_rank': scale_rank}
    else:
        raise ValueError('Invalid embed_type')
    return adjs


def calculate_mrr(term_dir,
                  embedding_path,
                  model_size=None,
                  model_layer=None,
                  strategy='end',
                  embed_type='contextual',
                  static_embedding_path=None):
    '''
    Calculate MRR for each layer of the model
        Parameters:
            term_dir: str
                Path to the directory of adj terms
            embedding_path: str
                Path to the adj embeddings
            model_size: str
                Size of the model
            model_layer: int
                Number of layers of the model
            strategy: str
                Strategy for computing dVec
            embed_type: str
                Type of embeddings
            static_embedding: dict
                Static embeddings
        Returns:
            mrr_res: dict
    '''
    adj_terms = read_scales(term_dir)
    if embed_type == 'static':
        static_embedding = read_all_adj_embeddings(static_embedding_path, embed_type)
    else:
        static_embedding = None
    adj_embeddings = read_all_adj_embeddings(embedding_path)
    selected_embeddings = generate_selected_embedding(adj_terms, adj_embeddings, model_size)
    scale_vec = compute_scale_vec(selected_embeddings,
                                  model_layer,
                                  strategy,
                                  embed_type,
                                  static_embedding)
    adjs = classify_scale(selected_embeddings,
                          scale_vec,
                          model_layer,
                          strategy,
                          embed_type,
                          static_embedding)
    mrr_res = {'demelo': [], 'crowd': [], 'wilkinson': []}
    if embed_type == 'contextual':
        for layer in range(1, model_layer+1):
            curr = adjs[layer]
            for data_name in curr.keys():
                mrr = 0
                for ranking in curr[data_name].values():
                    mrr += 1/(ranking['alignment_rank']+1)
                mrr /= len(curr[data_name])
                mrr_res[data_name].append(mrr)
    elif embed_type == 'static':
        curr = adjs
        for data_name in curr.keys():
            mrr = 0
            for ranking in curr[data_name].values():
                mrr += 1/(ranking['alignment_rank']+1)
            mrr /= len(curr[data_name])
            mrr_res[data_name].append(mrr)
    else:
        raise ValueError('Invalid embed_type')
    return mrr_res


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--term_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Path to adj terms')
    parser.add_argument('--embedding_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Path to contextual word embeddings')
    parser.add_argument('--embedding_type',
                        default='contextual',
                        type=str,
                        required=False,
                        help='Type of embeddings')
    parser.add_argument('--out_path',
                        default='.',
                        type=str,
                        required=False,
                        help='Path for output file')
    parser.add_argument('--static_embedding_path',
                        default=None,
                        type=str,
                        required=False,
                        help='Path to static word embeddings')
    parser.add_argument('--dvec_strategy',
                        default='end',
                        type=str,
                        required=False,
                        help='Strategy for computing dVec')

    args = parser.parse_args()
    file_name = args.embedding_path.split('/')[-1]
    model_name = file_name.split('_')[0]
    size = file_name.split('.')[0].split('_')[1]
    if size == 'base':
        layer = 12
    else:
        layer = 24

    if args.embedding_type == 'contextual':
        for i in range(10):
            random.seed(i)
            res_dic = calculate_mrr(args.term_path,
                                    args.embedding_path,
                                    size,
                                    layer,
                                    args.dvec_strategy)
            if not i:
                averaged_res = {'demelo': np.array([res_dic['demelo']]),
                                'wilkinson': np.array([res_dic['wilkinson']]),
                                'crowd': np.array([res_dic['crowd']])}
            else:
                for data_name in averaged_res.keys():
                    averaged_res[data_name] = np.concatenate((averaged_res[data_name],
                                                              np.array([res_dic[data_name]])),
                                                              axis=0)

        for data_name in averaged_res.keys():
            mean = np.average(averaged_res[data_name], axis=0)
            std = np.std(averaged_res[data_name], axis=0)
            max_val = np.max(mean)
            max_ind = np.argmax(mean)
            max_std = std[max_ind]
            print(f'In {data_name}, the best performance is: {max_val}, std is {max_std}')

        pickle.dump(averaged_res, open(f'{args.out_path}/{model_name}_{size}_{args.dvec_strategy}_alignment.pkl', "wb"))

    elif args.embedding_type == 'static':
        res_dic = calculate_mrr(args.term_path,
                                args.embedding_path,
                                strategy=args.dvec_strategy,
                                model_size=size,
                                model_layer=layer,
                                embed_type=args.embedding_type,
                                static_embedding_path=args.static_embedding_path)
        for data_name in res_dic.keys():
            mean = np.average(res_dic[data_name])
            std = np.std(res_dic[data_name])
            print(f'In {data_name}, the best performance is: {mean}, std is {std}')
        pickle.dump(res_dic, open(f'{args.out_path}/{model_name}_{args.dvec_strategy}_alignment.pkl', "wb"))

    else:
        raise ValueError('Invalid embed_type')


if __name__ == '__main__':
    main()
