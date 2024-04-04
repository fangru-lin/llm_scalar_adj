from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import json
import numpy as np
from copy import deepcopy
import argparse
from utils.utils import get_response_gpt4
import requests
import time
from bs4 import BeautifulSoup


def get_next_token(tokenizer, mask, model, model_name, text, device, top_k=5):
    '''
    Get the next token
        Parameters:
            tokenizer (AutoTokenizer): tokenizer for assessed model
            mask (str): mask token
            model (AutoModel): assessed model
            model_name (str): model name
            text (str): text
            device (str): device name
            top_k (int): top k
        Return:
            set: top k indices
    '''
    if 'bert' in model_name:
        # Tokenize input
        input_ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        masked_index = tokens.index(mask)

        # Predict all tokens
        with torch.no_grad():
            outputs = model(input_ids)
            predictions = outputs[0]

        probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
        top_k_indices = torch.topk(probs, top_k, sorted=True)[-1]
    else:
        encoded = tokenizer(text, return_tensors='pt').to(device)
        outputs = model.generate(input_ids=encoded.input_ids,
                                 attention_mask=encoded.attention_mask,
                                 return_dict_in_generate=True,
                                 pad_token_id=0,
                                 max_new_tokens=1,
                                 output_scores=True)

        # Predict all tokens
        top_k_indices = torch.topk(outputs.scores[0][0], top_k, sorted=True)[-1]

    return set([int(ind) for ind in top_k_indices])


def get_adj_ids(rankings, tokenizer, model_name):
    '''
    Get adj ids
        Parameters:
            rankings (dict): rankings
            tokenizer (AutoTokenizer): tokenizer for assessed model
            model_name (str): model name
        Return:
            id_dic (dict): dictionary that maps adjs to ids
            ranking_with_ids (dict): rankings with ids
    '''
    id_dic = dict()
    ranking_with_ids = deepcopy(rankings)
    for data_name, data in rankings.items():
        for ranking_name, ranking in data.items():
            ranking_with_ids[data_name][ranking_name] = set()
            for adjs in ranking:
                for adj in adjs:
                    if adj not in id_dic:
                        adj_ids = tokenizer(f' {adj}').input_ids
                        if 'flan' in model_name.lower():
                            if len(adj_ids) == 2:
                                id_dic[adj] = adj_ids[0]
                                ranking_with_ids[data_name][ranking_name].add(adj_ids[0])
                        elif 'bert' in model_name.lower():
                            if len(adj_ids) == 3:
                                id_dic[adj] = adj_ids[1]
                                ranking_with_ids[data_name][ranking_name].add(adj_ids[1])
                        elif 'falcon' in model_name.lower():
                            if len(adj_ids) == 1:
                                id_dic[adj] = adj_ids[0]
                                ranking_with_ids[data_name][ranking_name].add(adj_ids[0])
    return id_dic, ranking_with_ids


def assign_scales(rankings,
                  ranking_with_ids,
                  model,
                  tokenizer,
                  id_dic,
                  model_name,
                  pattern,
                  device):
    '''
    Assign scales
        Parameters:
            rankings (dict): rankings
            ranking_with_ids (dict): rankings with ids
            model (AutoModel): assessed model
            tokenizer (AutoTokenizer): tokenizer for assessed model
            id_dic (dict): dictionary that maps adjs to ids
            model_name (str): model name
            pattern (str): pattern
            device (str): device name
        Return:
            dict: accuracy dictionary
    '''
    if 'bert' in model_name:
        mask = tokenizer.mask_token
    acc_dic = {'demelo': 0, 'crowd': 0, 'wilkinson': 0}
    for data_name, data in rankings.items():
        total = 0
        correct = 0
        for ranking_name, ranking in data.items():
            all_adjs = ranking_with_ids[data_name][ranking_name]
            for adjs in ranking[:-1]:
                for adj in adjs:
                    if all_adjs == set([id_dic.get(adj, -1)]):
                        continue
                    total += 1

                    if 'bert' in model_name:
                        text = f'{adj} {pattern} {mask},'
                    else:
                        text = f'{adj} {pattern} '

                    top_5 = get_next_token(tokenizer,
                                           mask,
                                           model,
                                           model_name,
                                           text,
                                           device,
                                           top_k=5)
                    overlap = top_5.intersection(all_adjs)

                    if len(overlap) > 1:
                        correct += 1
                    elif len(overlap) == 1 and overlap != set([id_dic.get(adj, -1)]):
                        correct += 1
        acc = correct/total
        acc_dic[data_name] = acc

    return acc_dic


def get_all_non_extreme_adjs(rankings):
    '''
    Get all non extreme adjs
        Parameters:
            rankings (dict): rankings
        Return:
            set: non extreme adjs
    '''
    non_extreme_adjs = set()
    for data in rankings.values():
        for ranking in data.values():
            all_adjs = [adj for adjs in ranking[:-1] for adj in adjs]
            non_extreme_adjs.update(set(all_adjs))
    return non_extreme_adjs


def generate_scale_alignment_prompt_gpt4(non_extreme_adjs):
    '''
    Generate scale alignment prompt for GPT-4
        Parameters:
            non_extreme_adjs (set): non extreme adjs
        Return:
            list: prompts
    '''
    prompts = list()
    for adj in non_extreme_adjs:
        prompt = f'Do not provide explanations. Give five most likely words following the phrase: {adj.strip()} or even'
        prompts.append(prompt)
    return prompts


def prompt_gpt4_for_scale(rankings, model_name):
    '''
    Prompt GPT-4 for scale classification
        Parameters:
            rankings (dict): rankings
            model_name (str): model name
        Return:
            dict: scale alignment dictionary
    '''
    scale_alignment_dic = dict()
    non_extreme_adjs = get_all_non_extreme_adjs(rankings)
    scale_alignment_prompts = generate_scale_alignment_prompt_gpt4(non_extreme_adjs)
    for i, prompt in enumerate(scale_alignment_prompts):
        response = get_response_gpt4(prompt, model_name, max_tokens=100)
        print(response)
        scale_alignment_dic[non_extreme_adjs[i]] = response
    return scale_alignment_dic


def get_gpt4_scale_alignment_res(alignment_dic, rankings):
    '''
    Get GPT-4 scale alignment results
        Parameters:
            alignment_dic (dict): alignment dictionary
            rankings (dict): rankings
        Return:
            dict: results dictionary
    '''
    res_dic = dict()
    for data_name, data in rankings.items():
        total = 0
        correct = 0
        for ranking in data.values():
            all_adjs = set([adj.strip() for adjs in ranking for adj in adjs])
            for adjs in ranking[:-1]:
                for adj in adjs:
                    total += 1
                    scale_res = alignment_dic[adj.strip()]
                    intersect = set(scale_res).intersection(all_adjs)
                    if len(intersect) > 1:
                        correct += 1
                    elif len(intersect) == 1 and intersect != set([adj]):
                        correct += 1
        res_dic[data_name] = correct/total
    return res_dic


def get_membership_queries_ngram(membership_patterns, rankings):
    '''
    Get membership queries for ngram
        Parameters:
            membership_patterns (list): membership patterns
            rankings (dict): rankings
        Return:
            list: query list
    '''
    query_list = list()
    membership_adjs = set()
    for dataname, data in rankings.items():
        for rankings_name, ranking in data.items():
            words = [ii.strip() for i in ranking for ii in i]
            for word in words[:-1]:
                membership_adjs.add(word)
            # print(word_pairs)
    for pattern in membership_patterns:
        if ',' in pattern:
            continue
        for adj in membership_adjs:
            query_list.append(f'{adj} {pattern} *')
    return query_list


def get_ngrams_membership(query_list):
    '''
    Get ngrams membership
        Parameters:
            query_list (list): query list
        Return:
            dict: top 5 completions
    '''
    w = 1
    query = ','.join(query_list)
    params = dict(content=query, year_start=2019, year_end=2019,
                  corpus=37, smoothing=0) # use eng-2019 corpus

    req = requests.get('http://books.google.com/ngrams/graph', params=params)
    while 'Too Many Requests' in req.text:
        print(f'too many requests, will wait for {w} min')
        time.sleep(60*w)
        w += 1
        req = requests.get('http://books.google.com/ngrams/graph', params=params)
    parsed_html = BeautifulSoup(req.text)

    if parsed_html.find_all("script", {"id": "ngrams-data"}):
        freqs = json.loads(parsed_html.find_all("script", {"id": "ngrams-data"})[-1].text)
    else:
        return dict()

    freq_dic = dict()
    for freq in freqs:
        completion = freq['ngram'].split()[-1]
        if completion == '*':
            continue
        pattern = ' '.join(freq['ngram'].split()[:-1])
        if pattern not in freq_dic:
            freq_dic[pattern] = list()
        freq_dic[pattern].append((completion, freq['timeseries'][-1]))

    top5_dic = {'if not': dict(), 'or even': dict()}

    for phrase in freq_dic.keys():
        weak = phrase.split()[0]
        pattern = ' '.join(phrase.split()[1:])
        freq_dic[phrase].sort(key=lambda x: x[-1], reverse=True)
        top5_dic[pattern][weak] = set()
        for completion, freq in freq_dic[phrase][:5]:
            top5_dic[pattern][weak].add(completion)

    return top5_dic


def get_top5_completions(query_list):
    '''
    Get top 5 completions
        Parameters:
            query_list (list): query list
        Return:
            dict: top 5 completions
    '''
    top5_dic = {'if not': dict(), 'or even': dict()}
    i = 0
    while i < len(query_list):
        completions = get_ngrams_membership(query_list[i:i+12])
        i += 12
        for key, val in completions.items():
            for word, completion in val.items():
                top5_dic[key][word] = completion
    return top5_dic


def assign_scales_ngram(rankings,
                        top5_dic,
                        patterns):
    '''
    Assign scales for ngram
        Parameters:
            rankings (dict): rankings
            top5_dic (dict): top 5 completions
            patterns (list): patterns
        Return:
            dict: accuracy dictionary
    '''
    acc_dic = dict()
    for pattern in patterns:
        acc_dic[pattern] = {'demelo': 0, 'crowd': 0, 'wilkinson': 0}
        for data_name, data in rankings.items():
            total = 0
            correct = 0
            for ranking_name, ranking in data.items():
                all_adjs = set([adj.strip() for adjs in ranking for adj in adjs])
                for adjs in ranking[:-1]:
                    for adj in adjs:
                        total += 1

                        top_5 = top5_dic[pattern].get(adj, set())
                        overlap = top_5.intersection(all_adjs)

                        if len(overlap) > 1:
                            correct += 1
                        elif len(overlap) == 1 and overlap != set([adj]):
                            correct += 1
            acc = correct/total
            acc_dic[pattern][data_name] = acc

    return acc_dic


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name',
                        default=None,
                        type=str,
                        required=True,
                        help='Full model name')
    parser.add_argument('--device',
                        default='cuda',
                        type=str,
                        required=False,
                        help='Specify device')
    parser.add_argument('--rankings',
                        default=None,
                        type=str,
                        required=True,
                        help='Path for rankings')
    parser.add_argument('--output_path',
                        default='.',
                        type=str,
                        required=False,
                        help='Path for output file')

    args = parser.parse_args()

    with open(args.rankings) as f:
        rankings = json.load(f)

    if 'gpt' in args.model_name:
        scale_alignment_dic = prompt_gpt4_for_scale(rankings, args.model_name)
        with open('scale_alignment_dic.json', 'w') as f:
            json.dump(scale_alignment_dic, f)
        res_dic = get_gpt4_scale_alignment_res(scale_alignment_dic, rankings)
        return res_dic
    if 'ngram' in args.model_name:
        # ngram corpus does not encode punctuations so we only test two patterns
        membership_patterns = ['or even', 'if not']
        membership_queries = get_membership_queries_ngram(membership_patterns, rankings)
        top5_dic = {'if not': dict(), 'or even': dict()}
        completions = get_ngrams_membership(membership_queries)
        for key, val in completions.items():
            for word, completion in val.items():
                top5_dic[key][word] = completion
        with open(f'{args.output_path}/ngram_top5_dic.json', 'w') as f:
            json.dump(top5_dic, f)
        ngram_res = assign_scales_ngram(rankings,
                                        top5_dic,
                                        membership_patterns)
        with open(f'{args.output_path}/ngram_res.json', 'w') as f:
            json.dump(ngram_res, f)

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if 'bert' in model_name:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.to(args.device)
    elif 'flan' in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                      torch_dtype=torch.bfloat16,
                                                      device_map='auto')
    elif 'falcon' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map='auto')
    model.eval()
    patterns = ['or even', 'if not', ', or even', ', if not']

    id_dic, ranking_with_ids = get_adj_ids(rankings, tokenizer, model_name)
    acc_dic = {'demelo': list(), 'wilkinson': list(), 'crowd': list()}
    for pattern in patterns:
        print(model_name, pattern)
        pattern_acc = assign_scales(rankings,
                                    ranking_with_ids,
                                    model,
                                    tokenizer,
                                    id_dic,
                                    model_name,
                                    pattern,
                                    args.device)
        for key, val in pattern_acc.items():
            acc_dic[key] += [val]

    for key, val in acc_dic.items():
        res = np.array(val)
        print(f'Best performanc in {key}:')
        ind = res.argmax()+1
        print(f'template {ind},')
        best = '%.3f' % res.max()
        print(best)
        print('all results:')
        for r in res:
            r_print = '%.3f' % r
            print(r_print)


if __name__ == '__main__':
    main()
