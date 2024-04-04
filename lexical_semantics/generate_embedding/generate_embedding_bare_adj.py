from transformers import AutoTokenizer, AutoModel
from utils.utils import read_scales
import torch
import os
import random
import pickle
import argparse


def generate_adj_ids_dic(tokenizer, adj_terms):
    '''
    Generate tokenized ids for adjs.
        Parameters:
            tokenizer (AutoTokenizer): tokenizer for assessed model
            adj_terms (dict): adj terms in datasets
        Return:
            adj_ids_dic (dict): dictionary that maps adjs to input ids
    '''
    seps = tokenizer('')['input_ids']
    adj_ids_dic = dict()
    for scales in adj_terms.values():
        for scale in scales:
            for adj in scale:
                # add a blank space in front of all words to ensure identical token ids
                ids = tokenizer(f' {adj}')['input_ids']
                if len(seps) == 2:
                    adj_ids_dic[adj] = torch.tensor(ids[1:-1])
                elif len(seps) == 1:
                    if ids.index(seps[0]) == 0:
                        adj_ids_dic[adj] = torch.tensor(ids[1:])
                    else:
                        adj_ids_dic[adj] = torch.tensor(ids[:-1])
                else:
                    assert not len(seps)
                    adj_ids_dic[adj] = torch.tensor(ids)
    return adj_ids_dic


def match_word_position(adj_ids, input_ids):
    '''
    Match adj positions in input ids
        Parameters:
            adj_ids (list): tokenized adj ids
            input_ids (list): tokenized input ids
        Return:
            adj_st (int): start position
            adj_end (int): end position
    '''
    adj_len = len(adj_ids)
    for i in range(len(input_ids[0])-adj_len+1):
        if input_ids[0][i] == adj_ids[0]:
            if torch.equal(input_ids[0][i:i+adj_len], adj_ids):
                adj_st = i
                adj_end = i+adj_len-1
                return adj_st, adj_end


def pool_reps(states, pool_method):
    '''
    Mean/max pooling of target adjectives.
        Parameters:
            states (torch.tensor): hidden states of tokens of words
            pool_method (str): mean/max pooling
        Return:
            rep (torch.tensor): mean/max pooling results of tensors
    '''
    if pool_method == 'mean':
        rep = torch.mean(states, dim=-2)
        return rep.cpu()
    elif pool_method == 'max':
        rep = torch.max(states, dim=-2)
        return rep[0].cpu()
    else:
        raise Exception('wrong pooling method')


def extract_representation(adj_terms,
                           adj_ids_dic,
                           tokenizer,
                           model,
                           device):
    '''
    Extract representations for adjs.
        Parameters:
            adj_terms (dict): adj terms in datasets
            adj_ids_dic (dict): dictionary that maps adjs to input ids
            tokenizer (AutoTokenizer): tokenizer for assessed model
            model (AutoModel): assessed model
            device (str): device name
        Return:
            reps (dict): dictionary that maps adjs to representations
    '''
    reps = dict()

    with torch.no_grad():
        for scales in adj_terms.values():
            for scale in scales:
                if scale in reps:
                    continue
                reps[scale] = list()
                adjs = list(scale)
                for seed in range(10):
                    embedding_dic = dict()
                    instance = ' '.join(random.Random(seed).sample(adjs, len(adjs)))
                    # add a blank space in front of all words to ensure identical token ids
                    tok_inputs = tokenizer(f' {instance}')
                    input_ids = tok_inputs['input_ids']
                    input_ids = torch.tensor(input_ids).reshape(1, len(input_ids)).to(device)
                    outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)
                    hidden_states = outputs.encoder_hidden_states
                    for adj in adjs:
                        embedding_dic[adj] = {'mean': dict(), 'max': dict()}
                        adj_st, adj_end = match_word_position(adj_ids_dic[adj], input_ids.cpu())
                        for layer in range(1, len(hidden_states)):
                            embedding = hidden_states[layer][0][adj_st:adj_end+1]
                            for pool_method in embedding_dic[adj].keys():
                                embedding_dic[adj][pool_method][layer] = pool_reps(embedding,
                                                                                   pool_method)
                    reps[scale].append({'representation': embedding_dic})

    return reps


def generate_representation_dic(model_name,
                                adj_term_path,
                                device,
                                out_path):
    '''
    Generate dictionary for adj representation.
        Parameters:
            model_name (str): name for model
            adj_term_path (str): path for adj terms
            device (str): device name
            out_path (str): path for output_file
        Return:
            None
    '''
    adj_terms = read_scales(adj_term_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    adj_ids_dic = generate_adj_ids_dic(tokenizer, adj_terms)
    reps_dic = extract_representation(adj_terms,
                                      adj_ids_dic,
                                      tokenizer,
                                      model,
                                      device)
    # Create dir if non-existent
    os.makedirs(out_path, exist_ok=True)
    save_name = model_name.split('/')[-1]
    with open(f'{out_path}/{save_name}.pkl', 'wb') as f:
        pickle.dump(reps_dic, f)
    print('Word representation file saved!')


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path',
                        default='.',
                        type=str,
                        required=False,
                        help='Path for output file')
    parser.add_argument('--model',
                        default=None,
                        type=str,
                        required=True,
                        help='BERT or RoBERTa')
    parser.add_argument('--size',
                        default=None,
                        type=str,
                        required=True,
                        help='Specify model size')
    parser.add_argument('--device',
                        default='mps',
                        type=str,
                        required=False,
                        help='Specify device')
    parser.add_argument('--data_folder',
                        default='.',
                        type=str,
                        required=False,
                        help='Path for data folder')

    args = parser.parse_args()

    generate_representation_dic(args.model,
                                args.data_folder,
                                args.device,
                                args.out_path)


if __name__ == '__main__':
    main()
