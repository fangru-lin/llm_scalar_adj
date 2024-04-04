from transformers import pipeline, AutoTokenizer, AutoModel
import os
import json
import math
import argparse
import torch
import random
from copy import deepcopy
from itertools import combinations
from utils.utils import get_response_gpt4


def read_ranking_scales(dirname):
    '''
    Read all comparable pairs from the same scale from scal term files in dirname;
    keep track of which file each pair comes from
        Parameters:
            dirname: str
                Path to the directory
        Returns:
            rankings: dict
    '''
    termsfiles = [os.path.join(dirname, f) for f in os.listdir(dirname)]
    rankings = dict()
    for tf in termsfiles:
        if '.DS_Store' in tf:
            continue
        tf_clean = os.path.basename(tf).replace('.terms', '')
        ranking = []
        with open(tf, 'r') as fin:
            for line in fin:
                __, w = line.strip().split('\t')
                ranking.append(w.split('||'))

        rankings[tf_clean] = ranking
    return rankings


def load_rankings(data_dir, datanames=["demelo", "crowd", "wilkinson"]):
    '''
    Load all ranking scales from the data_dir
        Parameters:
            data_dir: str
                Path to the data directory
            datanames: list
                List of data names
        Returns:
            rankings: dict
    '''
    rankings = dict()
    for dataname in datanames:
        rankings[dataname] = read_ranking_scales(data_dir + dataname + "/gold_rankings/")

    return rankings


def calculate_prob(string, tokenizer, pipe, model, device='cuda'):
    '''
    Calculate the probability of a string (estimation)
        Parameters:
            string: str
                Input string
            tokenizer: tokenizer
                Tokenizer
            pipe: pipeline
                Pipeline
            model: str
                Model name
            device: str
                Device
        Returns:
            float
    '''
    if 'bert' in model.lower():
        string_prob = 0
        tokens = tokenizer.convert_ids_to_tokens(tokenizer(string)['input_ids'][1:-1])
        for i in range(len(tokens)):
            mask = tokenizer.mask_token
            inputs = ' '.join(tokens[:i]+[mask]+tokens[i+1:])
            token_prob = pipe(inputs, targets=[tokens[i]])[0]['score']
            # convert to log prob to avoid underflow
            string_prob += math.log(token_prob)
        return string_prob
    elif 't5' in model.lower():
        input_ids = tokenizer(string, return_tensors = 'pt').input_ids.to(device)
        outputs = pipe(input_ids = input_ids, labels = input_ids)
        return -outputs.loss
    else:
        input_ids = tokenizer(string, padding = True, return_tensors='pt').input_ids.to(device)
        logits = pipe(input_ids=input_ids, labels=input_ids).logits
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1))
        loss_by_sequence = loss.view(logits.size(0), -1).mean(dim=1)
        return -loss_by_sequence


def compare_prob(pattern, w, s, tokenizer, pipe, model, device='cuda'):
    '''
    Compare the probability of two strings
        Parameters:
            pattern: list
                Pattern
            w: str
                Word
            s: str
                Word
            tokenizer: tokenizer
                Tokenizer
            pipe: pipeline
                Pipeline/Model
            model: str
                Model name
            device: str
                Device
        Returns:
            bool
    '''
    if len(pattern) > 1:
        correct_str = f'{pattern[0]} {w} {pattern[-1]} {s}'
        wrong_str = f'{pattern[0]} {s} {pattern[-1]} {w}'
    else:
        correct_str = f'{w} {pattern[-1]} {s}'
        wrong_str = f'{s} {pattern[-1]} {w}'

    correct_prob = calculate_prob(correct_str, tokenizer, pipe, model, device=device)
    wrong_prob = calculate_prob(wrong_str, tokenizer, pipe, model, device=device)
    return correct_prob > wrong_prob


def compare_prob_tie(pattern, w, s, tokenizer, pipe, model, device='cuda'):
    '''
    Compare the probability of two strings
        Parameters:
            pattern: list
                Pattern
            w: str
                Word
            s: str
                Word
            tokenizer: tokenizer
                Tokenizer
            pipe: pipeline
                Pipeline/Model
            model: str
                Model name
            device: str
                Device
        Returns:
            bool
    '''
    if len(pattern) > 1:
        correct_str = f'{pattern[0]} {w} {pattern[-1]} {s}'
        wrong_str = f'{pattern[0]} {s} {pattern[-1]} {w}'
    else:
        correct_str = f'{w} {pattern[-1]} {s}'
        wrong_str = f'{pattern[0]} {s} {pattern[-1]} {w}'
    correct_prob = calculate_prob(correct_str, tokenizer, pipe, model, device=device)
    wrong_prob = calculate_prob(wrong_str, tokenizer, pipe, model, device=device)
    return correct_prob == wrong_prob


def get_ranking(model, ppl, tokenizer, rankings, out_path, device='cuda'):
    '''
    Get ranking results
        Parameters:
            model: str
                Model name
            ppl: pipeline
                Pipeline/Model
            tokenizer: tokenizer
                Tokenizer
            rankings: dict
                Rankings
            out_path: str
                Output path
            device: str
                Device
        Returns:
            None
    '''
    all_res = dict()
    best_res = {'average':{'strategy': [], 'res':0, 'demelo':0, 'crowd': 0, 'wilkinson':0},
                'demelo':{'strategy': [], 'res':0},
                'wilkinson':{'strategy': [], 'res': 0},
                'crowd':{'strategy': [], 'res': 0}}
    for key, val in patterns.items():
        all_res[key] = list()
        for i, pattern in enumerate(val):
            print(key, pattern)
            # temp dict to save results
            temp = {'demelo': 0, 'crowd': 0, 'wilkinson': 0}
            for data_name, data in rankings.items():
                total = 0
                correct = 0
                for rank_name, ranking in data.items():
                    for i, words in enumerate(ranking):
                        for ii, word in enumerate(words):
                            if words[ii+1:]:
                                for tie in words[ii+1:]:
                                    total += 1
                                    correct += int(compare_prob_tie(pattern,
                                                                           word,
                                                                           tie,
                                                                           tokenizer,
                                                                           ppl,
                                                                           model,
                                                                           device=device))
                            for adjs in ranking[i+1:]:
                                for adj in adjs:
                                    total += 1
                                    if key == 'weak_strong':
                                        correct += int(compare_prob(pattern,
                                                                           word,
                                                                           adj,
                                                                           tokenizer,
                                                                           ppl,
                                                                           model,
                                                                           device=device))
                                    else:
                                        correct += int(compare_prob(pattern,
                                                                    adj,
                                                                    word,
                                                                    tokenizer,
                                                                    ppl,
                                                                    model,
                                                                    device=device))
                acc = correct/total
                temp[data_name] = acc
                if acc > best_res[data_name]['res']:
                    best_res[data_name]['strategy'] = [[key, i]]
                    best_res[data_name]['res'] = acc
                elif acc == best_res[data_name]['res']:
                    best_res[data_name]['strategy'].append([key, i])
            all_res[key].append(temp)
            avg_res = sum(temp.values())/3
            if avg_res > best_res['average']['res']:
                best_res['average']['res'] = avg_res
                best_res['average']['strategy'] = [[key, i]]
            elif avg_res == best_res['average']['res']:
                best_res['average']['strategy'].append([key, i])
    print(f'Best performance is {best_res['average']['res']}')
    print(f'Best strategies are {best_res['average']['strategy']}')
    model_name = model.split('/')[-1]
    with open(f'{out_path}/indirect_ranking_res_{model_name}', 'w') as f:
        json.dump(all_res, f)


def get_all_possible_pairs(rankings):
    '''
    Get all possible adj pairs from rankings
        Parameters:
            rankings: dict
                Rankings
        Returns:
            set
    '''
    pairs = set()
    for data_name, data in rankings.items():
        for ranking_name, ranking in data.items():
            all_adjs  = [adj for adjs in ranking for adj in adjs]
            all_pairs = combinations(all_adjs, 2)
            pairs.update(set(all_pairs))
    return pairs


def generate_indirect_ranking_prompt_gpt4(pairs):
    '''
    Generate indirect ranking prompts for GPT-4
        Parameters:
            pairs: set
                Pairs
        Returns:
            prompt_dic: dict
                Prompt dictionary
    '''
    prompt_dic = {'prompts':list(), 'gold':list()}
    for i, pair in enumerate(pairs):
        shuffled_pair = random.Random(i).sample([pair[0], pair[1]], 2)
        weak, strong = shuffled_pair
        if shuffled_pair == list(pair):
            prompt_dic['gold'].append('A')
        else:
            prompt_dic['gold'].append('B')
        prompt = f'Do not provide explanations. Which of the following phrases is more natural? Answer none if they are equally unnatural. \nA. not just {weak.strip()} but {strong.strip()}\nB. not just {strong.strip()} but {weak.strip()}'
        prompt_dic['prompts'].append(prompt)
    return prompt_dic


def prompt_gpt4_for_ranking(rankings, model_name):
    '''
    Prompt GPT-4 for ranking
        Parameters:
            rankings: dict
                Rankings
            model_name: str
                Model name
        Returns:
            ranking_res: dict
                Ranking results
    '''
    pairs = get_all_possible_pairs(rankings)
    ranking_prompts = generate_indirect_ranking_prompt_gpt4(pairs)
    ranking_res = deepcopy(ranking_prompts)
    ranking_res['answers'] = list()
    for i, prompt in enumerate(ranking_prompts['prompts']):
        # print(prompt)
        # return
        response = get_response_gpt4(prompt, model_name, max_tokens = 100)
        print(response)
        # return
        ranking_res['answers'].append(response)
    return ranking_res


def get_ranking_res_gpt4(ranking_response, rankings):
    '''
    Get ranking results for GPT-4
        Parameters:
            ranking_response: dict
                Ranking response
            rankings: dict
                Rankings
        Returns:
            res_dic: dict
                Result dictionary
    '''
    res_dic = dict()
    pair_wise_ranking = dict()
    for i, answer in enumerate(ranking_response['answers']):
        prompt = ranking_response['prompts'][i]
        choicea, choiceb = prompt.split('A. ')[-1].split('\nB. ')
        if ranking_response['gold'][i] == 'A':
            weak, strong = choicea.split('not just ')[-1].split(' but ')
        else:
            weak, strong = choiceb.split('not just ')[-1].split(' but ')
        if ranking_response['gold'][i] in answer:
            pair_wise_ranking[(weak.strip(), strong.strip())] = '<'
        elif 'none' in answer:
            pair_wise_ranking[(weak.strip(), strong.strip())] = '='
        else:
            pair_wise_ranking[(weak.strip(), strong.strip())] = '>'

        
    for data_name, data in rankings.items():
        total=0
        correct = 0
        for ranking in data.values():
            for i, adjs in enumerate(ranking):
                for ii, weak in enumerate(adjs):
                    if adjs[ii+1:]:
                        for tie in adjs[ii+1:]:
                            total+=1
                            if pair_wise_ranking[(weak.strip(), tie.strip())] == '=':
                                correct+=1
                    for strongs in ranking[i+1:]:
                        for strong in strongs:
                            total+=1
                            if pair_wise_ranking[(weak.strip(), strong.strip())] == '<':
                                correct+=1
        res_dic[data_name] = correct/total
    return res_dic


def get_ranking_ngram(rankings, out_path, pattern_dic, word_pairs):
    '''
    Get intensity ranking via google ngram
        Parameters:
            rankings: dict
                Rankings
            out_path: str
                Output path
            pattern_dic: dict
                Pattern dictionary
            word_pairs: list
                Word pairs
        Returns:
            None
    '''
    freq_dic = dict()
    for relation, patterns in pattern_dic.items():
        if relation not in freq_dic.keys():
            freq_dic[relation] = dict()
        for i, pattern in enumerate(patterns):
            if i<8 and relation=='weak_strong':
                continue
            pattern_key = '*'.join(pattern)
            if ',' not in pattern_key:
                freq_dic[relation][pattern_key] = dict()
            else:
                continue
            query_list = list()
            query_pair_list = list()
            for pair in word_pairs:
                w, s = pair
                ws = w+' '+pattern[0]+' '+s+' '.join(pattern[1:])
                sw = s+' '+pattern[0]+' '+w+' '.join(pattern[1:])
                if ws not in query_list:
                    query_pair_list.append((w,s))
                    query_list.append(ws)
                if sw not in query_list:
                    query_pair_list.append((s,w))
                    query_list.append(sw)
            ii=0
            freq_list = list()
            while ii<len(query_list):
                freqs = get_ngrams(query_list[ii:ii+12])
                freq_list+=freqs
                ii+=12
            for idx, freq in enumerate(freq_list):
                freq_dic[relation][pattern_key][query_pair_list[idx]] = freq

    all_res = dict()
    best_res = {'average':{'strategy': [], 'res':0, 'demelo':0, 'crowd': 0, 'wilkinson':0},
                'demelo':{'strategy': [], 'res':0},
                'wilkinson':{'strategy': [], 'res': 0},
                'crowd':{'strategy': [], 'res': 0}}
    for key, val in freq_dic.items():
        all_res[key] = list()
        for i, pattern in enumerate(val):
            print(key, pattern)
            pattern_key = pattern
            # temp dict to save results
            temp = {'demelo': 0, 'crowd': 0, 'wilkinson': 0}
            for data_name, data in rankings.items():
                total = 0
                correct = 0
                for ranking in data.values():
                    for i, words in enumerate(ranking):
                        for ii, word in enumerate(words):
                            if words[ii+1:]:
                                for tie in words[ii+1:]:
                                    total += 1
                                    correct += int(freq_dic[key][pattern_key][(word.strip(), tie.strip())] == freq_dic[key][pattern_key][(tie.strip(), word.strip())])
                            for adjs in ranking[i+1:]:
                                for adj in adjs:
                                    total += 1
                                    if key == 'weak_strong':
                                        correct += int(freq_dic[key][pattern_key][(word.strip(), adj.strip())] > freq_dic[key][pattern_key][(adj.strip(), word.strip())])
                                    else:
                                        correct += int(freq_dic[key][pattern_key][(word.strip(), adj.strip())] < freq_dic[key][pattern_key][(adj.strip(), word.strip())])
                acc = correct/total
                temp[data_name] = acc
                if acc > best_res[data_name]['res']:
                    best_res[data_name]['strategy'] = [[key, pattern]]
                    best_res[data_name]['res'] = acc
                elif acc == best_res[data_name]['res']:
                    best_res[data_name]['strategy'].append([key, pattern])
            all_res[key].append(temp)
            avg_res = sum(temp.values())/3
            if avg_res > best_res['average']['res']:
                best_res['average']['res'] = avg_res
                best_res['average']['strategy'] = [[key, pattern]]
            elif avg_res == best_res['average']['res']:
                best_res['average']['strategy'].append([key, pattern])
            print(temp)
    print(best_res)
    model_name = 'ngram'
    with open(f'{out_path}/indirect_ranking_res_{model_name}.json', 'w') as f:
        json.dump(all_res, f)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--term_path',
                        default=None,
                        type=str,
                        required=True,
                        help='File of scalar terms')

    parser.add_argument('--out_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Path for output file')

    parser.add_argument('--model',
                        default=None,
                        type=str,
                        required=True,
                        help='model name')
    
    parser.add_argument('--device',
                        default='cuda',
                        type=str,
                        required=False,
                        help='device')
    

    args = parser.parse_args()

    rankings = load_rankings(args.term_path+'/')

    patterns = {'weak_strong': [['but not'], ['and almost'], ['and even'], ['or even'], ['although not'],
                ['if not'], ['though not'], ['or almost'], [', and even'], [', or even'],
                [', or almost'], [', and almost'], [', though not'], [', although not'],
                [', but not'], [', if not'], ['not only', 'but'], ['not just', 'but'],
                ['even'], ['almost'], [', even'], [', almost']],
                'strong_weak': [['not', 'just'], ['not', ', just'], ['not', 'but just'],
                ['not', ', but just'], ['not', 'but still'], ['not', ', but still'],
                ['not', 'still'], ['not', ', still'], ['not', 'although still'],
                ['not', ', although still'], ['not', ', though still'], ['not', 'though still']]}

    if 'gpt' in args.model:
        ranking_res = prompt_gpt4_for_ranking(rankings, args.model)
        res_dic = get_ranking_res_gpt4(ranking_res, rankings)
        print(res_dic)
        with open(f'{args.out_path}/indirect_ranking_res_{args,model}.json', 'w') as f:
            json.dump(ranking_res, f)
        return res_dic
    if 'ngram' in args.model:
        word_pairs = get_all_possible_pairs(rankings)
        get_ranking_ngram(rankings, args.out_path, patterns, word_pairs)
        return
    if 'bert' in args.model.lower():
        ppl = pipeline('fill-mask', args.model, device=args.device)
    else:
        ppl = AutoModel.from_pretrained(args.model, dtype=torch.float16).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    get_ranking(args.model, ppl, tokenizer, rankings, args.out_path, args.device)


if __name__ == '__main__':
    main()
