import os
import sys
import time
import datetime
import random
import logging
import json
import argparse
from tqdm import tqdm
import pickle
import re

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, LoggingHandler
from sentence_transformers.util import cos_sim, batch_to_device

from msc_dataset import gen_retrieval_query, Message
from rt_chat_corpus import load_chat_examples, deduplicate_chat_corpus

order_corpus_source = 'data/orders.json'

def load_user_gender(gender_file):
    if gender_file is None:
        return {}
    if not os.path.isfile(gender_file):
        return {}

    genders = {}
    with open(gender_file, 'r') as fin:
        for line in fin:
            item = json.loads(line)
            genders[item['user_id']] = item['gender']
    return genders

def encode(model, texts, batch_size=16, show_progress_bar=False, to_numpy=False):
    return model.encode(texts,
            batch_size,
            show_progress_bar=show_progress_bar, 
            convert_to_numpy=to_numpy,
            convert_to_tensor=not to_numpy)

class BiEncoderScanRetrival(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        model = SentenceTransformer(args.model_name_or_path, device=args.device)
        device = torch.device(args.device)
        feat_dump_file = args.feat_dump
        if os.path.isfile(feat_dump_file):
            with open(feat_dump_file, 'rb') as fin:
                feat_pack = pickle.load(fin)
                feats = torch.tensor(feat_pack['feats'], device=device)
                corpus = feat_pack['corpus']
                genders = feat_pack['genders']
                logging.info('load encoded features from %s', feat_dump_file)
        else:
            genders = load_user_gender(args.genders)
            logging.info('load genders completed, size:%s', len(genders))
            corpus = load_chat_examples(args.corpus, genders)
            corpus = deduplicate_chat_corpus(corpus)
            ans_corpus = [i[4] for i in corpus]
            feats = encode(model, ans_corpus, self.batch_size, show_progress_bar=True)
            # dump
            with open(feat_dump_file, 'wb') as fout:
                pickle.dump(dict(
                        feats=feats.cpu().numpy(),
                        corpus=corpus,
                        genders=genders,
                ), fout)
                logging.info('save encoded features to %s', feat_dump_file)

        self.corpus = corpus
        self.genders = genders
        self.feats = feats
        self.model = model
        self.device = device
        self.corpus_size = len(corpus)
        self.hist_wind = 10
        logging.info('load corpus and features. size=%s', self.corpus_size)

    def retrieve(self, msc, topk=10, batch_size=None, select_gender=None,
            exclude_userids=[], exclude_starids=[], include_starids=[], user_limit=3):
        """
            msc: chat history, [(is_user, msg)]
        """
        batch_size = batch_size or self.batch_size
        msc_msgs = [Message(is_use, msg) for is_use, msg in msc[-self.hist_wind:]]
        query = gen_retrieval_query(msc_msgs)
        logging.debug(query)
        qs_feat = encode(self.model, [query], 1)
        scores = torch.zeros(self.corpus_size, device=self.device)
        for i in range(0, self.corpus_size, batch_size):
            batch_scores = cos_sim(qs_feat, self.feats[i:i+batch_size])
            scores[i:i+batch_size] = batch_scores[0]

        # filter and topk
        rst_score, rst_idx = [], []
        ok_cnt = 0
        user_cnt = {}
        sort_idx = torch.argsort(-scores)
        for i in range(self.corpus_size):
            si = sort_idx[i]
            user_id, star_id, gender = self.corpus[si][:3]
            if user_id in exclude_userids:
                continue
            if star_id in exclude_starids:
                continue
            if self.genders and select_gender:
                if gender != select_gender:
                    continue
            if include_starids and (star_id not in include_starids):
                continue

            if user_id in user_cnt:
                if user_cnt[user_id] >= user_limit:
                    continue
                else:
                    user_cnt[user_id] += 1
            elif user_limit > 0:
                user_cnt[user_id] = 1
            ok_cnt += 1
            if ok_cnt > topk:
                break

            rst_score.append(scores[si].item())
            rst_idx.append(si.item())

        return rst_score, rst_idx

    def rerank(self, msc, answers, batch_size=None):
        batch_size = batch_size or self.batch_size
        msc_msgs = [Message(is_use, msg) for is_use, msg in msc]
        query = gen_retrieval_query(msc_msgs)
        logging.debug(query)
        texts = [query] + answers
        feats = encode(self.model, texts, batch_size)
        qs_feat = feats[0:1]
        answer_size = len(answers)
        scores = torch.zeros(answer_size, device=self.device)
        for i in range(0, answer_size, batch_size):
            batch_scores = cos_sim(qs_feat, feats[i+1:i+1+batch_size])
            scores[i:i+batch_size] = batch_scores[0]

        sort_idx = torch.argsort(-scores)
        return scores[sort_idx].cpu().numpy(), sort_idx.cpu().numpy()

def setup_logger(log_path, name=None, log_level=logging.DEBUG):
    if not name:
        logger = logging.getLogger()
        name = 'api_svr'
    else:
        logger = logging.getLogger(name=name)
    logger.setLevel(log_level)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_path:
        save_dir = os.path.dirname(log_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(log_path, mode='a+')
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

class Greeting(object):
    def __init__(self, greeting_corpus, time_zone=None):
        self.greetings = [
            ['hi, there'],
            ["hello, how are you?"],
            ["How can I help you today"],
            ["Hello how can I help you?"],
        ]
        if os.path.isfile(greeting_corpus):
            with open(greeting_corpus, 'r') as fin:
                self.greetings = json.load(fin)['all']

    def __call__(self):
        choice = random.choice(self.greetings)
        return choice


def get_args():
    """
    get up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', default='data/retrieval.log', type=str, required=False, help='default: data/retrieval.log')
    parser.add_argument('-m', '--model_name_or_path', default='bie_distilbert_ckps.best', type=str, required=False,
                        help='model path, default: bie_distilbert_ckps.5-15')
    parser.add_argument('--seed', type=int, default=42, help='default: 42')
    parser.add_argument('--corpus', type=str, default=order_corpus_source, help=f'default: {order_corpus_source}')
    parser.add_argument('--greetings', type=str, default='data/greetings.json', help=f'default: data/greetings.json')
    parser.add_argument('--genders', type=str, default='data/psychic.users.json.20220517-1456.json',
            help='default: data/psychic.users.json.20220517-1456.json')
    parser.add_argument('--device', type=str, default='cuda:0', help="default: cuda:0")
    parser.add_argument('--batch_size', type=int, default=32, help="default: 32")
    parser.add_argument('--fp16', action="store_true", default=False, help="user fp16")
    parser.add_argument('-t', '--tokenizer_path', default=None, type=str, required=False, help='tokenized path')
    parser.add_argument('-f', '--feat_dump', default='./data/rt_feats.pickle', type=str, required=False,
            help='encoded feature of the docs, default:./data/rt_feats.pickle')
    return parser.parse_args()

def log_print(log_str, fout):
    print(log_str)
    print(log_str, file=fout)

def log_input(prefix, fout):
    in_str = input(prefix)
    print(f'{prefix} {in_str}', file=fout)
    return in_str


def main():
    args = get_args()
    setup_logger(args.log_path)
    retrival = BiEncoderScanRetrival(args)

    num_return_sequences = 5
    USER_NAME = 'User'
    AI_NAME = 'Advisor'
    greet_gen = Greeting(args.greetings)
    with open(args.log_path, 'a') as fout:
        while True:
            msc = []
            try:
                # greet
                greets = greet_gen()
                for g in greets:
                    msc.append((False, g))
                    log_print(f"<< {AI_NAME}: {g}", fout)
                while True:
                    in_str = log_input(f'>> {USER_NAME}:', fout)
                    if in_str.strip():
                        msc.append((True, in_str))
                    start = time.time()
                    top_score, top_idx = retrival.retrieve(msc, topk=num_return_sequences)
                    end = time.time()
                    answers = [retrival.corpus[i][4] for i in top_idx]
                    rerank_score, rerank_idx = retrival.rerank(msc, answers)
                    for i, (score, idx, rscore) in enumerate(zip(top_score, top_idx, rerank_score)):
                        answer = retrival.corpus[idx][4]
                        if i == 0:
                            msc.append((False, answer))
                        log_print('-' * 20, fout)
                        if num_return_sequences < 2:
                            log_print(f"<< {AI_NAME}: {answer} \t{score} \t {rscore}", fout)
                        else:
                            log_print(f"<< {AI_NAME} top{i+1}: {answer} \t{score} \t {rscore}", fout)

                    fout.flush()
            except KeyboardInterrupt as err:
                log_print(f'end this chat', fout)

if __name__ == '__main__':
    main()
