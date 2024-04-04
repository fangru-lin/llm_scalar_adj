import os
import numpy as np
import random
import io
import requests
import time
import json
import numpy as np
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


@retry(wait=wait_random_exponential(min=1, max=50), stop=stop_after_attempt(6))
def chat_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def load_vectors(fname):
    '''
    Load static embeddings
        Parameters:
            fname (str): file name
        Return:
            data (dict): embeddings
    '''
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(tokens[1:], dtype=float)
    print('vectors loaded')
    return data


def read_scales(dirname):
    '''
    Read scales from three datasets
    Modified from Gari Soler & Apidianki (2020)
        Parameters:
            dirname (str): directory of the datasets
        Return:
            adj_terms (dict): adjectives in datasets
    '''
    adj_terms = dict()
    for f in os.listdir(dirname):
        # Skip system file
        if f == '.DS_Store':
            continue
        folder = os.path.join(dirname, f)
        adj_terms[f] = list()
        termsfiles = [os.path.join(folder+'/terms', termfile)
                      for termfile in os.listdir(folder + '/terms') if termfile != '.DS_Store']
        for tf in termsfiles:
            terms = []
            with open(tf, 'r') as fin:
                for line in fin:
                    __, w = line.strip().split('\t')
                    terms.append(w)

            adj_terms[f].append(tuple(terms))
        adj_terms[f] = set(adj_terms[f])
    return adj_terms


def read_all_adj_embeddings(path,
                            embed_type='contextual'):
    '''
    Read all adjective embeddings from a file
        Parameters:
            path: str
                Path to the file
            embed_type: str
                Type of embeddings
        Returns:
            adjs: dict
    '''
    if embed_type == 'contextual':
        adjs = []
        with (open(path, "rb")) as openfile:
            while True:
                try:
                    adjs.append(pickle.load(openfile))
                except EOFError:
                    break
            openfile.close()
        return adjs[0]
    elif embed_type == 'static':
        adjs = load_vectors(path)
        return adjs


def generate_selected_embedding(adj_terms,
                                adj_embeddings,
                                model_size='base'):
    '''
    Generate selected adjective embeddings
        Parameters:
            adj_terms: dict
                Adjective terms
            adj_embeddings: dict
                Adjective embeddings
            model_size: str
                Size of the model

        Returns:
            embeddings: dict
    '''
    embeddings = {'demelo': dict(),
                  'crowd': dict(),
                  'wilkinson': dict()}
    if model_size == 'base':
        layer_num = 12
    else:
        layer_num = 24
    for group in adj_embeddings.keys():
        group_embedding = dict()
        random_sample = random.sample(range(10), len(group))
        for i in range(len(group)):
            sent_id = random_sample[i]
            adj = group[i]
            group_embedding[adj] = dict()
            for layer in range(layer_num+1):
                layer_rep = adj_embeddings[group][sent_id]['representations'][adj][layer]
                group_embedding[adj][layer] = layer_rep
        for key in embeddings.keys():
            if group in adj_terms[key]:
                embeddings[key][group] = group_embedding

    return embeddings


def get_scale_words(scale):
    words = []
    for w in scale:
        words.extend(w.split(" || "))
    return words


def get_response_gpt4(prompt, model_name="gpt-4-0613", max_tokens=1):
    response = chat_with_backoff(model=model_name,
                                 messages=[{"role": "user", "content": prompt}],
                                 temperature=0,
                                 max_tokens=max_tokens,
                                 top_p=1,
                                 frequency_penalty=0,
                                 presence_penalty=0)
    return response.choices[0].message.content


def get_ngrams(query_list):
    w = 1
    query = ','.join(query_list)
    params = dict(content=query, year_start=2019, year_end=2019,
                  corpus=37, smoothing=0) # use eng-2019 corpus

    req = requests.get('http://books.google.com/ngrams/graph', params=params)
    while 'Too Many Requests' in req.text:
        print(f'too many requests, will wait for {w} min')
        time.sleep(60*w)
        w+=1
        req = requests.get('http://books.google.com/ngrams/graph', params=params)
    res = list()
    parsed_html = BeautifulSoup(req.text)
    
    if parsed_html.find_all("script", {"id": "ngrams-data"}):
        freqs = json.loads(parsed_html.find_all("script", {"id": "ngrams-data"})[-1].text)
    else:
        res = [0 for _ in range(len(query_list))]

    j=0
    for i, freq in enumerate(freqs):
        while j<len(query_list) and freq['ngram'] != query_list[j]:
            res.append(0)
            j+=1
        if j>=len(query_list):
            break
        if freq['ngram'] == query_list[j]:
            res.append(freq['timeseries'][-1])
            j+=1
    if len(res)<len(query_list):
        res+=[0 for _ in range(len(query_list)-len(res))]
    return res