#import wget
import argparse
from dataclasses import dataclass
import random
from typing import Dict, List

import nltk
import numpy as np
import torch
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import BartTokenizer, BertTokenizer, RobertaTokenizerFast,T5TokenizerFast
from src.dataset_classes import DatasetObject, Features

boolean = bool

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_base', '-mb', default="t5-base",
                        help='The type of transformer architecture')
    parser.add_argument('--output_dir', '-output_dir',
                        help='Location of where the trained model is saved')
    parser.add_argument('--run_id','-run_id',type=str,default='',
                        help='Id for the running')
    parser.add_argument('--eval_steps','-eval_steps',type=int,default=1000)
    parser.add_argument('--learning_rate','-learning_rate',type=float,default=5e-5)
    parser.add_argument('--max_squad_size', default=80000, type=int,
                        help='Maximum number of samples from the squad dataset')
    parser.add_argument('--max_seq_len','-max_seq_len',default=512,type=int)

    parser.add_argument('--seed', default=10, type=int, help='Random seed')
    parser.add_argument('--lr_scheduler_type','-lr_scheduler_type', default='cosine')
    parser.add_argument('--weight_decay','-weight_decay',type=float, default=0.3)
    parser.add_argument('--warmup_ratio','-warmup_ratio',type=float, default=0.21)
    parser.add_argument('--num_train_epochs','-num_train_epochs',type=int, default=4)
    
    parser.add_argument('--save_total_limit','-save_total_limit',type=int, default=1)
    parser.add_argument('--per_device_train_batch_size','-per_device_train_batch_size',type=int, default=16)
    parser.add_argument('--per_device_eval_batch_size','-per_device_eval_batch_size',type=int, default=16)
    parser.add_argument('--verbose','-verbose',action='store_true')
    #warmup_ratio save_total_limit per_device_eval_batch_size
    args = parser.parse_args()
    return args

def process_extra(extra_data):
  data = extra_data.values
  dataset = []
  for idx,dat in tqdm(enumerate(data)):
    fact,answer_sentence = dat[3],dat[2]
    dob_q= DatasetObject(question=dat[1],context=dat[0].strip(),answer=dat[2],answer_sentence=answer_sentence.strip(),fact=fact,task='<generate_questions> ',task_id=1)
    dataset.append(dob_q)

    dob_a= DatasetObject(question=dat[1],context=dat[0].strip(),answer=dat[2],answer_sentence=answer_sentence.strip(),fact=fact,task='<generate_answers> ',task_id=2)
    dataset.append(dob_a)

  return dataset

def selectContext(articles, answer_index,n=4):
    vals = list(ngrams(range(len(articles)),n))
    for l in vals:
        if answer_index in l:
            return l
def buildFact(answer, context):
    answer = answer.strip()
    context = context.strip()
    contexts = sent_tokenize(context)
    new_contexts = ' '  # contexts[0]
    answer_index = [idx for idx, s in enumerate(contexts) if answer in s]
    if len(answer_index) < 1:
        return None
    answer_index = answer_index[0]
    
    if len(contexts) < 5:
        new_contexts = ' '.join(contexts)
        return new_contexts, contexts[answer_index]
    
    selected_context_idxs = selectContext(contexts,answer_index,n=random.choice([4,3]))
    new_contexts = ' '.join([contexts[sid] for sid in selected_context_idxs])
    return new_contexts, contexts[answer_index]
    
    
    if answer_index == 0:
        new_contexts = ' '.join(contexts[:5])
    elif answer_index == len(contexts)-1:
        new_contexts = ' '+contexts[answer_index-2]+' ' + \
            contexts[answer_index-1]+' '+contexts[answer_index]
    else:
        new_contexts = ' '+ contexts[answer_index-1]+' ' + \
            contexts[answer_index]+' '+contexts[answer_index+1]

    return new_contexts, contexts[answer_index]


def questionGeneratorDataset(data, n_samples=20000,verbose=False):
    data_qg = data.sample(n_samples).values
    dataset = []
    for idx, dat in tqdm(enumerate(data_qg)) if verbose else enumerate(data_qg):
        response = buildFact(dat[4], dat[3])
        if response is not None:
            fact, answer_sentence = response
        else:
            continue
        dob_ = DatasetObject(question=dat[0], context=dat[3].strip(
        ), answer=dat[4], answer_sentence=answer_sentence.strip(), fact=fact, task='<generate_questions> ', task_id=1)
        dataset.append(dob_)
    return dataset
# <generate_questions> <generate_answers>


def answerGeneratorDataset(data, n_samples=20000,verbose=False):
    data_qg = data.sample(n_samples).values
    dataset = []
    for idx, dat in tqdm(enumerate(data_qg)) if verbose else enumerate(data_qg):
        response = buildFact(dat[4], dat[3])
        if response is not None:
            fact, answer_sentence = response

        else:
            continue
        fact, answer_sentence = buildFact(dat[4], dat[3])
        dob_ = DatasetObject(question=dat[0], context=dat[3].strip(
        ), answer=dat[4], answer_sentence=answer_sentence, fact=fact, task='<generate_answers> ', task_id=2)
        dataset.append(dob_)
    return dataset

def process_extra(extra_data):
  data = extra_data.values
  dataset = []
  for idx,dat in tqdm(enumerate(data)):
    fact,answer_sentence = dat[3],dat[2]
    dob_q= DatasetObject(question=dat[1],context=dat[0].strip(),answer=dat[2],answer_sentence=answer_sentence.strip(),fact=fact,task='<generate_questions> ',task_id=1)
    dataset.append(dob_q)

    dob_a= DatasetObject(question=dat[1],context=dat[0].strip(),answer=dat[2],answer_sentence=answer_sentence.strip(),fact=fact,task='<generate_answers> ',task_id=2)
    dataset.append(dob_a)

  return dataset

def setuptokenizer(model_base='bert-base-uncased', additional_tokens=[], special_tokens=[],):
    if 'bert-' in model_base:
        tokenizer = BertTokenizer.from_pretrained(model_base)
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
    elif 'bart' in model_base:
        tokenizer = BartTokenizer.from_pretrained(model_base)
    elif 't5-' in model_base:
        tokenizer = T5TokenizerFast.from_pretrained(model_base)
    elif 'roberta-' in model_base:
        tokenizer = RobertaTokenizerFast.from_pretrained(model_base)
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
    else:
        return None
    tokenizer.add_tokens(additional_tokens)
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    return tokenizer


def pad_seq(seq: List[np.ndarray], max_batch_len: int, pad_value: int, verbose=False) -> List[int]:
    if len(seq) > max_batch_len:
        seq = seq.to(torch.long).unsqueeze(0)[:, :max_batch_len]
        return seq
    pads = torch.from_numpy(np.array([pad_value]*(max_batch_len - len(seq))))
    out = torch.concat([seq, pads], -1).to(torch.long).unsqueeze(0)
    return out


@dataclass
class SmartCollator:
    pad_token_id: int
    label_pad_token_id: int = -100
    is_gpt: boolean = False
    max_len: int = 512

    def __call__(self, batch: List[Features]) -> Dict[str, torch.Tensor]:
        batch_inputs: List = list()
        batch_attention_masks: List = list()
        decoder_attention_mask: List = list()
        labels: List = list()
        max_size = min([max([len(ex.input_ids)
                       for ex in batch]), self.max_len])

        max_size_output = min([max([len(ex.labels) for ex in batch]), self.max_len])

        for item in batch:
            batch_inputs += [pad_seq(item.input_ids,
                                     max_size, self.pad_token_id)]
            batch_attention_masks += [
                pad_seq(item.attention_mask, max_size, 0)]

            if not self.is_gpt:
                decoder_attention_mask += [
                    pad_seq(item.decoder_attention_mask, max_size_output, 0)]
            labels += [pad_seq(item.labels, max_size_output,
                               self.label_pad_token_id)]
        if not self.is_gpt:
            return dict(
                input_ids=torch.concat(batch_inputs, 0),
                attention_mask=torch.concat(batch_attention_masks, 0),
                labels=torch.concat(labels, 0),
                decoder_attention_mask=torch.concat(decoder_attention_mask, 0),)
        else:
            return dict(
                input_ids=torch.concat(batch_inputs, 0),
                attention_mask=torch.concat(batch_attention_masks, 0),
                labels=torch.concat(labels, 0),)
