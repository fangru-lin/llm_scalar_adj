import openai
import json
import math
import torch
import string
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


@retry(wait=wait_random_exponential(min=1, max=50), stop=stop_after_attempt(6))
def chat_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def get_yes_no_prob(prompt, yes_id, no_id, model, tokenizer):
    '''
    Get yes and no probabilities for a given prompt for non-gpt models.
        Parameters:
            prompt (str): prompt
            yes_id (int): yes token id
            no_id (int): no token id
            model (AutoModel): model for assessment
            tokenizer (AutoTokenizer): tokenizer for assessment
        Return:
            orig_probs (list): original probabilities
            weighted_probs (list): weighted probabilities
    '''
    encoded = tokenizer(prompt, return_tensors='pt').to('cuda')
    outputs = model.generate(input_ids=encoded.input_ids,
                             attention_mask=encoded.attention_mask,
                             return_dict_in_generate=True,
                             pad_token_id=0,
                             max_new_tokens=1,
                             output_scores=True)
    orig_yes, orig_no = outputs.scores[0].softmax(1)[0, [yes_id, no_id]]
    weighted_yes, weighted_no = orig_yes/(orig_yes+orig_no), orig_no/(orig_yes+orig_no)
    # in orig_probs the probs of yes and no are taken from the softmax distribution from all tokens in the vocab
    orig_probs = [float(orig_yes), float(orig_no)]
    # in weighted_probs prob_yes+prob_no=1
    weighted_probs = [float(weighted_yes), float(weighted_no)]
    return orig_probs, weighted_probs


def get_yes_no_ids(tokenizer, model_name):
    '''
    Get yes and no ids for different models.
        Parameters:
            tokenizer (AutoTokenizer): tokenizer for assessed model
            model_name (str): model name
        Return:
            yes_id (int): yes token id
            no_id (int): no token id
    '''
    if 't5' in model_name:
        yes_id = int(tokenizer('yes', return_tensors='pt').to('cuda')['input_ids'][0][0])
        no_id = int(tokenizer('no', return_tensors='pt').to('cuda')['input_ids'][0][0])
    else:
        yes_id = int(tokenizer('. Yes', return_tensors='pt').to('cuda')['input_ids'][0][1])
        no_id = int(tokenizer('. No', return_tensors='pt').to('cuda')['input_ids'][0][1])
    return yes_id, no_id


def get_calib_coeff(template, yes_id, no_id, model, tokenizer):
    '''
    Get calibration coefficient for non-gpt models.
        Parameters:
            template (list): template for calibration
            yes_id (int): yes token id
            no_id (int): no token id
            model (AutoModel): model for assessment
            tokenizer (AutoTokenizer): tokenizer for assessment
        Return:
            W (np.array): calibration coefficient
    '''
    content_free_inputs = ["N/A", "", "[MASK]"]
    probs = [0, 0]
    for i, c_input in enumerate(content_free_inputs):
        for turn_2 in content_free_inputs[:i]+content_free_inputs[i+1:]:
            pseudo_prompt = template[0]+c_input+template[1]+turn_2+template[2]
            yes_bias, no_bias = get_yes_no_prob(model, tokenizer, pseudo_prompt, yes_id, no_id)[-1]
            probs[0] += yes_bias
            probs[1] += no_bias
    p_y = np.array(probs)
    p_y = p_y/np.sum(p_y)
    W = np.linalg.inv(np.identity(2) * p_y)
    return W


def get_conf_dic(template_dic,
                 model,
                 tokenizer,
                 yes_id,
                 no_id,
                 prompt_dic):
    '''
    Get confidence dictionary for non-gpt models.
        Parameters:
            template_dic (dict): template dictionary
            model (AutoModel): model for assessment
            tokenizer (AutoTokenizer): tokenizer for assessment
            yes_id (int): yes token id
            no_id (int): no token id
            prompt_dic (dict): prompt dictionary
        Return:
            conf_dic (dict): confidence dictionary
    '''
    conf_dic = dict()

    for data_name in ['rx', 'ng', 'pvt']:
        template = template_dic[data_name]
        W = get_calib_coeff(template, yes_id, no_id)
        conf_dic[data_name] = dict()
        conf_dic[data_name] = {'orig_yes':list(),
                              'orig_no':list(),
                              'weighted_yes':list(),
                              'weighted_no':list(),
                              'calibed_yes':list(),
                              'calibed_no':list()}
        for i, prompts in enumerate(prompt_dic[data_name]['prompts']):
            conf_dic[data_name]['orig_yes'].append([])
            conf_dic[data_name]['orig_no'].append([])
            conf_dic[data_name]['weighted_yes'].append([])
            conf_dic[data_name]['weighted_no'].append([])
            conf_dic[data_name]['calibed_yes'].append([])
            conf_dic[data_name]['calibed_no'].append([])
            for ii, prompt in enumerate(prompts):
                [orig_yes, orig_no], [weighted_yes, weighted_no] = get_yes_no_prob(model, tokenizer, prompt, yes_id, no_id)
                conf_dic[data_name]['orig_yes'][-1].append(orig_yes)
                conf_dic[data_name]['orig_no'][-1].append(orig_no)
                conf_dic[data_name]['weighted_yes'][-1].append(weighted_yes)
                conf_dic[data_name]['weighted_no'][-1].append(weighted_no)
                weighted_probs = np.array([weighted_yes, weighted_no])
                probs = W*weighted_probs
                calibed_yes = probs[0][0]/((probs).sum())
                conf_dic[data_name]['calibed_yes'][-1].append(calibed_yes)
                conf_dic[data_name]['calibed_no'][-1].append(1-calibed_yes)
    return conf_dic


def get_yes_or_no_gpt4(prompt, model_name="gpt-4-0613", max_tokens=1):
    '''
    Get yes or no response for GPT-4.
        Parameters:
            prompt (str): prompt
            model_name (str): model name
            max_tokens (int): max tokens
        Return:
            response (str): response
    '''
    response = chat_with_backoff(model=model_name,
                                 messages=[{"role": "user", "content": prompt+' Only answer yes or no.'}],
                                 temperature=0,
                                 max_tokens=max_tokens,
                                 top_p=1,
                                 frequency_penalty=0,
                                 presence_penalty=0)
    return response.choices[0].message.content


def prompt_gpt4(model_name: str,
                prompt_dic: dict):
    '''
    Get GPT diversity responses.
        Parameters:
            model_name (str): model name
            prompt_dic (dict): prompt dictionary

        Returns:
            response_dic (dict): confidence dictionary
    '''
    response_dic = dict()

    for data_name in ['rx', 'ng', 'pvt']:
        response_dic[data_name] = list()
        for prompt in prompt_dic[data_name]['prompts']:
            response = get_yes_or_no_gpt4(prompt, model_name)
            response_dic[data_name].append(response)

    return response_dic


def get_f1_gpt4(gpt4_dic, gold_dic):
    '''
    Get f1 score for GPT-4.
        Parameters:
            gpt4_conf_dic (dict): GPT-4 confidence dictionary
            gold_dic (dict): gold dictionary
        Returns:
            res_dic (dict): f1 score dictionary
    '''
    res_dic = dict()
    total_f1 = 0
    for data_name, data in gpt4_dic.items():
        temp_res = list()
        for responses in data:
            yes_count = 0
            for response in responses:
                if 'yes' in response.lower():
                    yes_count += 1
                    if yes_count >= len(responses)/2:
                        break
                else:
                    assert 'no' in response.lower()
            res = (yes_count >= len(responses)/2)
            temp_res.append(res)
        f1 = f1_score(gold_dic[data_name], temp_res, average='macro')
        total_f1 += f1
        res_dic[data_name] = f1
    return res_dic


def eval_diversity(pred_dic, gold_endorsement, strategy):
    '''
    Calculate yes/no prediction for non-GPT models.
        Parameters:
            pred_dic (dict): prediction dictionary
            gold_endorsement (list): gold endorsement
            strategy (str): strategy name
        Returns:
            res_dic (dict): f1 score dictionary
    '''
    res_dic = dict()
    for data_name in pred_dic.keys():
        gold = list()
        for i in gold_endorsement[data_name]:
            if i >= 0.5:
                gold.append('yes')
            else:
                gold.append('no')
        conf_list = pred_dic[data_name][strategy]
        judgements = list()
        for i, confs in enumerate(conf_list):
            count = 0
            if type(confs) is list:
                for conf in confs:
                    if 'no' not in strategy:
                        if conf >= 0.5:
                            count += 1
                    else:
                        if conf < 0.5:
                            count += 1
                if count/40 >= 0.5:
                    judgements.append('yes')
                else:
                    judgements.append('no')
            else:
                if confs >= 0.5:
                    judgements.append('yes')
                else:
                    judgements.append('no')
        res_dic[data_name] = f1_score(gold, judgements, average='macro')
    return res_dic


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name',
                        default=None,
                        type=str,
                        required=True,
                        help='Full model name')

    parser.add_argument('--api_key',
                        default=None,
                        type=str,
                        required=False,
                        help='OpenAI api key if using gpt-4')

    parser.add_argument('--out_path',
                        default='.',
                        type=str,
                        required=False,
                        help='Path for output file')

    parser.add_argument('--device',
                        default='cuda',
                        type=str,
                        required=False,
                        help='Device for model')

    parser.add_argument('--prompt_dic',
                        default=None,
                        type=str,
                        required=True,
                        help='Prompt dictionary path')

    args = parser.parse_args()

    template_dic = {'rx': ['Mary: ', ' Would you conclude from this that Mary thinks ', '?'],
                    'ng': ['Mary says: ', ' Woud you conclude from this that, according to Mary, ', '?'],
                    'pvt': ['Imagine that your friend Mary says, \"', '\" Would you conclude from this that Mary thinks ','?']}

    if 'gpt-4' in args.model_name:
        openai.api_key = args.api_key
        with open(args.prompt_dic, 'r') as f:
            prompt_dic = json.load(f)
        gpt4_dic = prompt_gpt4(args.model_name, prompt_dic)
        with open(f'{args.out_path}/gpt4_output.json', 'w') as f:
            json.dump(gpt4_dic, f)
        with open(f'{args.out_path}/gpt4_eval.json', 'w') as f:
            json.dump(get_f1_gpt4(gpt4_dic, prompt_dic['gold_endorsement']), f)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(args.device)

    yes_id, no_id = get_yes_no_ids(tokenizer, args.model_name)
    with open(args.prompt_dic, 'r') as f:
        prompt_dic = json.load(f)

    conf_dic = get_conf_dic(template_dic,
                            model,
                            tokenizer,
                            yes_id,
                            no_id,
                            prompt_dic)
    save_name = args.model_name.split('/')[-1]
    with open(f'{args.out_path}/{save_name}_conf_dic.json', 'w') as f:
        json.dump(conf_dic, f)

    for strategy in ['weighted_yes', 'weighted_no', 'calibed_yes', 'calibed_no']:
        eval_dic = eval_diversity(conf_dic, prompt_dic['gold_endorsement'], strategy)
        with open(f'{args.out_path}/{save_name}_{strategy}_eval.json', 'w') as f:
            json.dump(eval_dic, f)
