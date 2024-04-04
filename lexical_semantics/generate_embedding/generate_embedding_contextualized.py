'''
This script generate contextualised word embeddings for base and large models
of BERT and RoBERTa.
'''

import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
import pickle
from copy import deepcopy
import argparse


def aggregate_reps(reps_list,
                   hidden_size,
                   pooling_method='mean'):
    # This function averages representations of
    # a word that has been split into wordpieces.
    reps = torch.zeros([len(reps_list), hidden_size])
    for i, wrep in enumerate(reps_list):
        w, rep = wrep
        reps[i] = rep

    if len(reps) > 1:
        if pooling_method == 'mean':
            reps = torch.mean(reps, axis=0)
        elif pooling_method == 'max':
            reps = torch.max(reps, axis=0)[0]
        elif pooling_method == 'min':
            reps = torch.min(reps, axis=0)[0]
        else:
            raise ValueError("Pooling method not recognised")
    reps = reps.view(hidden_size)

    return reps.cpu()


def special_tokenization(sentence, tokenizer):
    map_ori_to_token = []
    start, end = tokenizer.batch_decode(tokenizer('')['input_ids'])
    tok_sent = [start]

    for orig_token in sentence.split():
        current_tokens_idx = [len(tok_sent)]
        curr_token = tokenizer.tokenize(orig_token)  # tokenize
        tok_sent.extend(curr_token)  # add to my new tokens
        if len(curr_token) > 1:  # if the new token has been 'wordpieced'
            extra = len(curr_token) - 1
            for i in range(extra):
                # list of new positions of the target word
                current_tokens_idx.append(current_tokens_idx[-1]+1)
        map_ori_to_token.append(tuple(current_tokens_idx))

    tok_sent.append(end)

    return tok_sent, map_ori_to_token


def extract_representations(infos,
                            tokenizer,
                            model,
                            device):
    reps = []

    model.eval()
    with torch.no_grad():
        for info in infos:
            tok_sent = info['tokenized_sentence']
            ids = [tokenizer.convert_tokens_to_ids(tok_sent)]
            input_ids = torch.tensor(ids).to(device)
            outputs = model(input_ids)
            hidden_states = outputs[2]
            positions = info["position"]

            reps_for_this_instance = dict()
            for i, w in enumerate(info["tokenized_sentence"]):
                if i in positions:
                    # all layers
                    for layer in range(len(hidden_states)):
                        if layer not in reps_for_this_instance:
                            reps_for_this_instance[layer] = []
                        embedding = hidden_states[layer][0][i].cpu()
                        reps_for_this_instance[layer].append((w, embedding))
            reps.append(reps_for_this_instance)

    return reps


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--sentence_path',
                        default='./ukwac_selected_scalar_sentences.pkl',
                        type=str,
                        required=False,
                        help='File of context sentences')
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
    parser.add_argument('--pooling_method',
                        default='mean',
                        type=str,
                        required=False,
                        help='Pooling method for wordpieces')

    args = parser.parse_args()

    data = pickle.load(open(args.sentence_path, "rb"))
    infos = []
    device = args.device

    if args.model == 'BERT':
        tokenizer = AutoTokenizer.from_pretrained(f"bert-{args.size}-uncased")
        config = AutoConfig.from_pretrained(f'bert-{args.size}-uncased',
                                            output_hidden_states=True)
        model = AutoModel.from_pretrained(f'bert-{args.size}-uncased',
                                          config=config)
        model.to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"roberta-{args.size}")
        config = AutoConfig.from_pretrained(f'roberta-{args.size}',
                                            output_hidden_states=True)
        model = AutoModel.from_pretrained(f'roberta-{args.size}',
                                          config=config)
        model.to(device)

    for scale in data:
        for instance in data[scale]:
            if '' in instance['sentence_words']:
                print(instance['sentence_words'])
            for scaleword in scale:
                cinstance = deepcopy(instance)
                sentence_words = list(cinstance["sentence_words"][:])

                # Replace a by an and viceversa if necessary
                quantifier = sentence_words[cinstance["position"]-1]
                if quantifier == "a" and scaleword[0] in 'aeiou':
                    sentence_words[cinstance["position"]-1] = "an"
                elif quantifier == "an" and scaleword[0] not in 'aeiou':
                    sentence_words[cinstance["position"]-1] = "a"

                # Replace original adjective by current adjective
                sentence_words[cinstance["position"]] = scaleword
                cinstance["position"] = [cinstance["position"]]

                sentence = " ".join(sentence_words)
                tokenization = special_tokenization(sentence, tokenizer)
                tokenized_sentence, mapp = tokenization
                current_positions = cinstance['position']

                if len(current_positions) == 1:
                    # list of positions (might have been split into wordpieces)
                    position = mapp[cinstance['position'][0]]
                elif len(current_positions) > 1:
                    position = []
                    for p in current_positions:
                        position.extend(mapp[p])

                cinstance["tokenized_sentence"] = tokenized_sentence
                cinstance["position"] = position
                cinstance["scale"] = scale
                cinstance["lemma"] = scaleword
                infos.append(cinstance)

    # EXTRACTING REPRESENTATIONS
    reps = extract_representations(infos,
                                   tokenizer,
                                   model,
                                   device)

    for rep, instance in zip(reps, infos):
        scale = instance["scale"]
        lemma = instance["lemma"]
        for ins2 in data[scale]:
            if ins2["sentence_words"] == instance["sentence_words"]:
                if "representations" not in ins2:
                    ins2["representations"] = dict()
                if lemma not in ins2["representations"]:
                    ins2["representations"][lemma] = dict()
                for layer in rep:
                    reps = aggregate_reps(rep[layer], model.config.hidden_size, args.pooling_method)
                    ins2['representations'][lemma][layer] = reps

    pickle.dump(data, open(f'{args.out_path}/{args.model}_{args.size}_embedding.pkl', "wb"))


if __name__ == '__main__':
    main()
