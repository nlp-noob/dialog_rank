import sys, os
import argparse
import json
import logging
from tqdm import tqdm
import numpy as np
import random
import re
import math
from collections import namedtuple

from torch.utils.data import Dataset
from transformers import AutoTokenizer

USER_TOKEN = "<User>"
AI_TOKEN = "<Advisor>"

Message = namedtuple('Message', ['sender', 'message'])  # sender: true from User
#CONV_PREFIX = "This is a coversation between a user called {user} who has questions about his/her love and finance and life also wants some predictions about their future and a advisor called {advisor} who has clairvoyance that and see the future and help the user to get the answer of  his/her question."
CONV_PREFIX = "This is a coversation between a customer called {user} and a advisor called {advisor}"


def get_file_list(path, ext):
    files = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            if filename.endswith(tuple(ext)):
                files.append(apath)

    return files

def sess_split_with_overlap(sess_cnt, max_sess_size, overlap_sess_size):
    split_idxs = [[i, min(i+max_sess_size, sess_cnt)] for i in range(0, sess_cnt, (max_sess_size - overlap_sess_size))]
    if len(split_idxs) > 1:
        s, e = split_idxs[-1]
        # merge last small batch
        if (e - s) <= 2 * overlap_sess_size:
            split_idxs[-2][1] = e
            split_idxs = split_idxs[:-1]
    return split_idxs

def sess_split(sess_cnt, max_sess_size, min_sess_size):
    avg_sess_size = (max_sess_size + min_sess_size) // 2
    split_idxs = [[i, min(i+avg_sess_size, sess_cnt)] for i in range(0, sess_cnt, avg_sess_size)]
    if len(split_idxs) > 1:
        # merge last small batch
        s, e = split_idxs[-1]
        if (e - s) <= min_sess_size:
            split_idxs[-2][1] = e
            split_idxs = split_idxs[:-1]
    return split_idxs

def print_distribute(desc, size_arr):
    print(f'{desc} distribution: size: {len(size_arr)} mean: {np.mean(size_arr)}, max: {np.max(size_arr)}, min: {np.min(size_arr)}, mediam: {np.median(size_arr)}, p95: {np.percentile(size_arr, 95)}, p90: {np.percentile(size_arr, 90)}, p80: {np.percentile(size_arr, 80)}')

def load_msc(order_file, msc_cnt_thresh=5, first_orders=5, first_msc_thresh=0): 
    with  open(order_file, 'r') as fin: 
        data = json.load(fin) 
    mscs = []
    stars = []
    order_infos = []
    msc_cnt = []
    ptn = re.compile(r'\n{2,}')
    for key, orders in tqdm(data.items()):
        # sort order by created
        orders = sorted(orders, key=lambda x: x['created_at'])
        orders = orders[:first_orders]
        for oi, order in enumerate(orders):
            star_id = order['star_id']
            msc = order['msc']
            info = order.get('order_info', {})
            # add order first label
            if oi == 0:
                info['is_first'] = True
            else:
                info['is_first'] = False

            if len(msc) < msc_cnt_thresh:
                continue
            if first_msc_thresh > 0:
                msc = msc[:first_msc_thresh]

            # preporcess
            def _preprocess_msg(msg_text):
                # replace \n+ to \n
                msg_text = ptn.sub('\n', msg_text)
                return msg_text

            msc = [Message(i[0], _preprocess_msg(i[1])) for i in msc]
            msc_cnt.append(len(msc))
            mscs.append(msc)
            stars.append(star_id)
            order_infos.append(info)
    print_distribute('origin msc', msc_cnt)
    return mscs, stars, order_infos

def gen_retrieval_query(msc):
    prompt_text = CONV_PREFIX.format(advisor=AI_TOKEN, user=USER_TOKEN)
    prompt_text += '\n\n'

    ptn = re.compile(r'\n{2,}')
    query = prompt_text
    # add history
    for msg in msc:
        if msg.sender:
            prefix = f'{USER_TOKEN}:'
        else:
            prefix = f'{AI_TOKEN}:'
        query += prefix + msg.message + '\n\n'
    # add advisor prefix
    query += f'{AI_TOKEN}:'
    return query

def gen_retrieval_text(msc, start, end):
    query = gen_retrieval_query(msc[start:end-1])
    assert len(msc) >= end
    msg = msc[end-1]
    assert not msg.sender
    paragraph = msg.message

    from sentence_transformers import InputExample
    return InputExample(texts=[query, paragraph])

def gen_cls_query(msc):
    prompt_text = CONV_PREFIX.format(advisor=AI_TOKEN, user=USER_TOKEN)
    prompt_text += '\n\n'

    ptn = re.compile(r'\n{2,}')
    query = prompt_text
    # add history
    for msg in msc:
        if msg.sender:
            prefix = f'{USER_TOKEN}:'
        else:
            prefix = f'{AI_TOKEN}:'
        query += prefix + msg.message + '\n\n'

    return query

def gen_cls_text(msc, start, end):
    query = gen_cls_query(msc[start:end-1])
    # role prefix
    assert len(msc) >= end
    msg = msc[end-1]
    label = 0 if msg.sender else 1 # 0 for user, 1 for advisor

    return query, label

class MSCRetrievalDataset(Dataset):
    """
        Multi-Turm dataset
    """
    def __init__(self, mscs, max_sess_size, min_sess_size, is_training=True):
        self.is_training = is_training
        self.mscs = mscs
        self.mscs_size = len(mscs)
        example_idx = []  # [(msc_idx, msc_adv_idx)]
        for msc_idx, msc in enumerate(mscs):
            msc_size = len(msc)
            example_idx.extend([(msc_idx, i) for i in range(msc_size) if not msc[i].sender])

        if not self.is_training:
            test_min_sess_thresh = 0  # 2 -> 0, eval score: 0.73 -> 0.78
            if test_min_sess_thresh > 0:
                example_idx = [(msc_idx, adv_idx) for msc_idx, adv_idx in example_idx if adv_idx >= test_min_sess_thresh]
            sess_size = math.ceil(0.5 * (max_sess_size + min_sess_size))
            examples = []
            for msc_idx, adv_idx in example_idx:
                end = adv_idx + 1
                start = max(0, end - sess_size)
                examples.append(gen_retrieval_text(self.mscs[msc_idx], start, end))
            self.examples = examples

        self.example_idx = example_idx
        self.max_sess_size, self.min_sess_size = max_sess_size, min_sess_size

    def __getitem__(self, index):
        if not self.is_training:
            return self.examples[index]

        msc_idx, adv_idx = self.example_idx[index]
        sess_size = random.randint(self.min_sess_size, self.max_sess_size)
        end = adv_idx + 1
        start = max(0, end - sess_size)
        return gen_retrieval_text(self.mscs[msc_idx], start, end)

    def shuffle(self):
        if not self.is_training:
            return
        random.shuffle(self.example_idx)

    def __len__(self):
        return len(self.example_idx)

class MTClfDataset(Dataset):
    """
        Multi-Turm dataset
    """
    def __init__(self, mscs, max_sess_size, min_sess_size, tokenizer, is_training=True):
        self.is_training = is_training
        self.mscs = mscs
        self.mscs_size = len(mscs)
        example_idx = []  # [(msc_idx, cls_pos_idx)]
        sess_size = math.ceil(0.5 * (max_sess_size + min_sess_size))
        self.tokenizer = tokenizer

        for msc_idx, msc in enumerate(mscs):
            msc_size = len(msc)
            msc_repeat_cnt = max(msc_size // sess_size, 1)
            cls_idxs = random.choices(list(range(msc_size)), k=msc_repeat_cnt)

            example_idx += [(msc_idx, -1 if is_training else i) for i in cls_idxs]

        if not self.is_training:
            test_min_sess_thresh = 2  # test for 0 or 2
            if test_min_sess_thresh > 0:
                example_idx = [(msc_idx, cls_idx) for msc_idx, cls_idx in example_idx if cls_idx >= test_min_sess_thresh]
            examples = []
            for msc_idx, cls_idx in example_idx:
                end = cls_idx + 1
                start = max(0, end - sess_size)
                text, label = gen_cls_text(self.mscs[msc_idx], start, end)
                if self.tokenizer is not None:
                    example = dict(self.tokenizer(text, truncation=True))
                    example['label'] = label
                else:
                    example = (text, label)
                examples.append(example)
            self.examples = examples

        self.example_idx = example_idx
        self.max_sess_size, self.min_sess_size = max_sess_size, min_sess_size

    def __getitem__(self, index):
        if not self.is_training:
            return self.examples[index]

        msc_idx, cls_idx = self.example_idx[index]
        msc = self.mscs[msc_idx]
        msc_len = len(msc)

        if cls_idx < 0:
            cls_idx = random.randint(0, msc_len-1)

        sess_size = random.randint(self.min_sess_size, self.max_sess_size)
        end = cls_idx + 1
        start = max(0, end - sess_size)
        text, label = gen_cls_text(self.mscs[msc_idx], start, end)
        if self.tokenizer is not None:
            example = dict(self.tokenizer(text, truncation=True))
            example['label'] = label
            return example
        else:
            return text, label

    def shuffle(self):
        if not self.is_training:
            return
        random.shuffle(self.example_idx)

    def __len__(self):
        return len(self.example_idx)

class MSCTokenizer:
    def __init__(self, star_list, tokenizer=None, concat=False):
        self.star_list = star_list
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained('data/dialogbert_tokenizer/')
        self.max_token_size = self.tokenizer.model_max_length
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        add_ids = self.tokenizer.additional_special_tokens_ids
        self.resp_token_id = add_ids[0]
        self.usr_token_id   = add_ids[1]
        self.advisor_id     = add_ids[2]
        self.advisor_ids    = add_ids[3:]
        self.concat = concat

    def _gen_ids(self, msc, star=None, is_response=False):
        ids = [self.cls_token_id]
        if is_response:
            ids += [self.resp_token_id]  # difference from context
        if not star or not self.star_list or star not in self.star_list:
            advisor_id = self.advisor_id
        else:
            idx = self.star_list.index(star)
            advisor_id = self.advisor_ids[idx]

        for msg in msc:
            if isinstance(msg, list):
                if msg[0]:
                    prefix = [self.usr_token_id]
                else:
                    prefix = [advisor_id]
                ids += prefix + self.tokenizer.encode(msg[1], add_special_tokens=False)
            else:
                if msg.sender:
                    prefix = [self.usr_token_id]
                else:
                    prefix = [advisor_id]
                ids += prefix + self.tokenizer.encode(msg.message, add_special_tokens=False)
        return ids

    def gen_ids(self, msc, star=None, is_response=False):
        skip = 2
        start_idx = 0
        ids = self._gen_ids(msc[start_idx:], star, is_response)
        while start_idx < len(msc) -1:
            if len(ids) <= self.max_token_size:
                break
            start_idx += skip
            ids = self._gen_ids(msc[start_idx:], star, is_response)

        if len(ids) > self.max_token_size:
            return ids[-self.max_token_size:]
        return ids

    def gen_context_response(self, msc, start, end, star=None):
        context_ids = self.gen_ids(msc[start:end-1], star)
        resp_ids = self.gen_ids(msc[end-1:end], star, is_response=False)

        if self.concat:
            concat_ids = context_ids + resp_ids[1:]
            concat_ids = concat_ids[:self.max_token_size]
            return context_ids, resp_ids, concat_ids

        return context_ids, resp_ids

    def gen_resp_ids(self, resp, star):
        # for tag_data
        resp_ids = self.gen_ids([[False, resp]], star, is_response=False)
        return resp_ids

    def gen_context_ids(self, msc, window_size, star):
        # for tag_data
        end = len(msc)
        start = end - window_size
        context_ids = self.gen_ids(msc[start:end], star, is_response=False)
        return context_ids


class MSCBertDataset(Dataset):
    """
    Multi-Turm dataset
    """
    def __init__(self, mscs, stars, cluster_labels, star_list, max_sess_size, min_sess_size, tokenizer, is_training=True, concat=False):
        self.is_training = is_training
        self.mscs = mscs
        self.stars = stars
        self.mscs_size = len(mscs)
        self.msc_tokenizer = MSCTokenizer(star_list, tokenizer, concat=concat)
        example_idx = []  # [(msc_idx, msc_adv_idx)]
        self.cluster_label_dict = []

        for msc_idx, msc in enumerate(mscs):
            msc_size = len(msc)
            label_dict = {i: label for i, label in cluster_labels[msc_idx]}
            advisor_idx = [(msc_idx, i) for i in range(msc_size) if not msc[i].sender]
            # check label
            for _, i in advisor_idx:
                assert i in label_dict
            example_idx.extend(advisor_idx)
            self.cluster_label_dict.append(label_dict)

        if not self.is_training:
            test_min_sess_thresh = 2  # 2 -> 0, eval score: 0.73 -> 0.78
            if test_min_sess_thresh > 0:
                example_idx = [(msc_idx, adv_idx) for msc_idx, adv_idx in example_idx if adv_idx >= test_min_sess_thresh]
            sess_size = math.ceil(0.5 * (max_sess_size + min_sess_size))
            examples = []
            for msc_idx, adv_idx in tqdm(example_idx):
                end = adv_idx + 1
                start = max(0, end - sess_size)
                # cluster label
                label = self.cluster_label_dict[msc_idx][adv_idx]
                input_ids = self.msc_tokenizer.gen_context_response(self.mscs[msc_idx], start, end, self.stars[msc_idx])
                examples.append((*input_ids, label))
#               examples.append(self.msc_tokenizer.gen_context_response(self.mscs[msc_idx], start, end, None))
            self.examples = examples

        self.example_idx = example_idx
        self.max_sess_size, self.min_sess_size = max_sess_size, min_sess_size

    def __getitem__(self, index):
        if not self.is_training:
            return self.examples[index]

        msc_idx, adv_idx = self.example_idx[index]
        label = self.cluster_label_dict[msc_idx][adv_idx]
        sess_size = random.randint(self.min_sess_size, self.max_sess_size)
        end = adv_idx + 1
        start = max(0, end - sess_size)
        if random.random() > 0.5:
            input_ids = self.msc_tokenizer.gen_context_response(self.mscs[msc_idx], start, end, None)  # for default advisor
        else:
            input_ids = self.msc_tokenizer.gen_context_response(self.mscs[msc_idx], start, end, self.stars[msc_idx])

        return *input_ids, label

    def shuffle(self):
        if not self.is_training:
            return
        random.shuffle(self.example_idx)

    def __len__(self):
        if not self.is_training:
            return len(self.examples)
        return len(self.example_idx)


class MSCGptTokenizer:
    USER_TOKEN = "[User]"
    AI_TOKEN = "[Advisor]"
    AI_TOKENS = "[Advisor{i}]"
    # CONV_PREFIX = "This is a coversation between a user called {user} who has questions about his/her love and finance and life also wants some predictions about their future and a advisor called {advisor} who has clairvoyance that and see the future and help the user to get the answer of  his/her question. First, The advisor ask some infomation of the user and the user's POI, and the user reply the advisor their info. Second, The advisor ask what the user want to know, and the user tell the advisor the question and some stroies of the user and the user's POI. Then the advisor get some answers of the users question by taking meditation and using their clairvayance."
    CONV_PREFIX_FIRST = "This is a coversation between a user called {user} who has questions about his/her love and finance and life also wants some predictions about their future and a advisor called {advisor} who has clairvoyance that and see the future and help the user to get the answer of  his/her question. It's their first conversation."
    CONV_PREFIX_FREE3M = "This is a coversation between a user called {user} who has questions about his/her love and finance and life also wants some predictions about their future and a advisor called {advisor} who has clairvoyance that and see the future and help the user to get the answer of  his/her question. It's the frist time for the user to ask a professional consultant for advice."
    CONV_PREFIX_REPUR = "This is a coversation between a user called {user} who has questions about his/her love and finance and life also wants some predictions about their future and a advisor called {advisor} who has clairvoyance that and see the future and help the user to get the answer of  his/her question. They had communicated before about some issues."
    CONV_PREFIX = "This is a coversation between a user called {user} who has questions about his/her love and finance and life also wants some predictions about their future and a advisor called {advisor} who has clairvoyance that and see the future and help the user to get the answer of  his/her question."
    CONV_SEP = "\n\n"

    def __init__(self, star_list, tokenizer=None):
        self.star_list = star_list
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained('gpt2')
        self.max_token_size = self.tokenizer.model_max_length

    def _get_tokens(self, star=None):
        user = self.USER_TOKEN
        if not star or not self.star_list or star not in self.star_list:
            advisor = self.AI_TOKEN
        else:
            idx = self.star_list.index(star)
            advisor = self.AI_TOKENS.format(i=idx)
        return user, advisor

    def _get_prefix(self, info, is_general=False):
        if is_general:
            return self.CONV_PREFIX

        is_free3m = info.get('is_free3m', 0) or info.get('is_3m_free', 0)
        if is_free3m:
            return self.CONV_PREFIX_FREE3M

        is_first = info.get('is_first', 0)
        if is_first:
            return self.CONV_PREFIX_FIRST

        return self.CONV_PREFIX_REPUR

    def _gen_text(self, msc, info, user, advisor, is_general=False):
        text = self._get_prefix(info, is_general).format(advisor=advisor, user=user)
        text += self.CONV_SEP

        for msg in msc:
            if msg.sender:
                text += f'{user}:'
            else:
                text += f'{advisor}:'

            text += msg.message + self.CONV_SEP
        return text

    def gen_prompt_text(self, msc, info, star=None, user_turn=False):
        user, advisor = self._get_tokens(star)
        text = self._gen_text(msc, user, info, advisor)
        if user_turn:
            text += f'{user}:'
        else:
            text += f'{advisor}:'
        return text

    def gen_ids(self, msc, info, star=None, is_general=False):
        user, advisor = self._get_tokens(star)
        skip = 2
        start_idx = 0
        text = self._gen_text(msc[start_idx:], info, user, advisor, is_general)
        ids = self.tokenizer.encode(text)
        while start_idx < len(msc) -1:
            if len(ids) <= self.max_token_size:
                break
            start_idx += skip
            text = self._gen_text(msc[start_idx:], info, user, advisor, is_general)
            ids = self.tokenizer.encode(text)

        if len(ids) > self.max_token_size:
            return ids[-self.max_token_size:]
        return ids

class MSCGptDataset(Dataset):
    """
        Multi-Turm dataset for clm
    """
    def __init__(self, mscs, stars, infos, star_list, max_sess_size, min_sess_size, tokenizer=None, is_training=True):
        self.is_training = is_training
        self.mscs = mscs
        self.stars = stars
        self.infos = infos
        self.mscs_size = len(mscs)
        self.prompt_tokenizer = MSCGptTokenizer(star_list, tokenizer)
        self.max_sess_size, self.min_sess_size = max_sess_size, min_sess_size

        example_idx = []  # [(msc_idx, end_pos)]
        avg_sess_size = math.ceil(0.5 * (max_sess_size + min_sess_size))
        for msc_idx, msc in enumerate(mscs):
            msc_size = len(msc)
            splited_batches = sess_split(msc_size, max_sess_size, min_sess_size)
            example_idx.extend([(msc_idx, s, e) for s, e in splited_batches])

        if not self.is_training:
            examples = []
            for msc_idx, start, end in tqdm(example_idx):
                examples.append(self.prompt_tokenizer.gen_ids(self.mscs[msc_idx][start:end], self.infos[msc_idx], self.stars[msc_idx]))
                examples.append(self.prompt_tokenizer.gen_ids(self.mscs[msc_idx][start:end], self.infos[msc_idx], None))
            self.examples = examples

        self.example_idx = example_idx

    def __getitem__(self, index):
        if not self.is_training:
            return self.examples[index]

        msc_idx, s, e = self.example_idx[index]
        mid = (s + e) // 2
        sess_size = random.randint(self.min_sess_size, self.max_sess_size)
        start = max(0, mid - sess_size // 2)
        end = start + sess_size

        is_general = False
        if random.random() > 0.7:
            is_general = True

        if random.random() > 0.5:
            return self.prompt_tokenizer.gen_ids(self.mscs[msc_idx][start:end], self.infos[msc_idx], None, is_general=is_general)  # for default advisor
        else:
            return self.prompt_tokenizer.gen_ids(self.mscs[msc_idx][start:end], self.infos[msc_idx], self.stars[msc_idx], is_general=is_general)

    def select(self, selector):
        if not self.is_training:
            self.examples = [self.examples[i] for i in selector]
            return

        self.example_idx = [self.example_idx[i] for i in selector]

    def shuffle(self):
        if not self.is_training:
            return
        random.shuffle(self.example_idx)

    def __len__(self):
        if not self.is_training:
            return len(self.examples)
        return len(self.example_idx)

def load_star_list(list_file):
    star_list = []
    if not os.path.isfile(list_file):
        return star_list
    with open(list_file, 'r') as fin:
        for line in fin:
            star_id = line.strip()
            star_list.append(star_id)

    return star_list

# this is the deprecated version because of the is_training loading process is missing
def load_msc_bert_test_datasets(
        test_order_file,
        msc_cnt_thresh=5,
        first_orders=5,
        first_msc_thresh=0,
        max_sess_size=15,
        min_sess_size=5,
        tokenizer=None,
        data_type=None,
        star_list=None,
        ):
    test_mscs, test_stars, test_infos = load_msc(test_order_file, msc_cnt_thresh, first_orders, first_msc_thresh)
    star_list = star_list or list(set(test_stars))
    star_list = sorted(star_list)

    mscs_size = len(test_mscs)
    random_idx = list(range(mscs_size))
    rstat = random.getstate()
    random.seed(42)
    random.shuffle(random_idx)
    random.setstate(rstat)
    test_mscs = [test_mscs[random_idx[i]] for i in range(len(test_mscs))]
    test_stars = [test_stars[random_idx[i]] for i in range(len(test_stars))]
    test_infos = [test_infos[random_idx[i]] for i in range(len(test_infos))]

    test_label_file = test_order_file[:-4] + "label.json"

    with open(test_label_file, 'r') as fin:
        test_cluster_label = json.load(fin)  #[ [(msc_idx, label), ],  ]
        test_cluster_label = [test_cluster_label[random_idx[i]] for i in range(len(test_cluster_label))]


    return MSCBertDataset(test_mscs, test_stars, test_cluster_label, star_list, max_sess_size, min_sess_size, tokenizer, is_training=False, concat=True)


def load_tagged_tri_dataset(data_path):
    with open(data_path, "r") as jf:
        tri_data = json.load(jf)
    dataset = []
    for cont_idx, data_item in enumerate(tri_data):
        cont_id = data_item["cont_id"]
        mlm_id = data_item["mlm_id"]
        for id_idx, p_resp_id in enumerate(data_item["p_resp_id"]):
            p_cluster_label = data_item["p_resp_cluster_idx"][id_idx]
            if id_idx >= len(data_item["n_resp_id"]):
                n_resp_id = data_item["n_resp_id"][id_idx % (len(data_item["n_resp_id"]) - 1)]
                n_cluster_label = data_item["n_resp_cluster_idx"][id_idx % (len(data_item["n_resp_id"]) - 1)]
            else:
                n_resp_id = data_item["n_resp_id"][id_idx]
                n_cluster_label = data_item["n_resp_cluster_idx"][id_idx]
            dataset.append(
                (cont_id[:510], p_resp_id[:510], n_resp_id[:510], mlm_id[:510], p_cluster_label, n_cluster_label, cont_idx)
            )
    return dataset


def load_tri_dataset(data_path):
    # tagged_data["cont"].append(cont_id)
    # tagged_data["resp"].append(resp_id)
    # tagged_data["mlm"].append(mlm)
    # tagged_data["p_index"].append(other_cluster_resp_index_list)
    # tagged_data["n_index"].append(rank_negative_resp_index_list)
    # tagged_data["dup_n_id"].append(advisor_sentences_in_cont)
    # tagged_data["p_cluster_label_list"].append(other_p_cluster_label_list)
    # tagged_data["n_cluster_label_list"].append(other_n_cluster_label_list)
    jf = open(data_path, "r")
    tri_data = json.load(jf)
    jf.close()

    filtered_tri_data = {
        "cont_id": [],
        "resp_id": [],
        "mlm_id": [],
        "p_index": [],
        "n_index": [],
        "cluster_label": [],
        "index":[],
        "p_cluster_label_list": [],
        "n_cluster_label_list": [],
    }

    index_to_cluster_label = tri_data["info"]["index_to_cluster_label"]

    # filter out of range data
    for i in range(len(tri_data["cont"])):
        cluster_label = tri_data["info"]["index_to_cluster_label"][str(i)]
        mlm_id = tri_data["mlm"][i]
        if len(mlm_id) > 510:
            filtered_tri_data["cont_id"].append(tri_data["cont"][i][:512])
            filtered_tri_data["resp_id"].append(tri_data["resp"][i][:512])
            filtered_tri_data["mlm_id"].append(tri_data["mlm"][i][:512])
            filtered_tri_data["p_index"].append(tri_data["p_index"][i])
            filtered_tri_data["n_index"].append(tri_data["n_index"][i])
            filtered_tri_data["cluster_label"].append(index_to_cluster_label[str(i)])
            filtered_tri_data["index"].append(i)
            filtered_tri_data["p_cluster_label_list"].append(tri_data["p_cluster_label_list"][i])
            filtered_tri_data["n_cluster_label_list"].append(tri_data["n_cluster_label_list"][i])
        else:
            filtered_tri_data["cont_id"].append(tri_data["cont"][i])
            filtered_tri_data["resp_id"].append(tri_data["resp"][i])
            filtered_tri_data["mlm_id"].append(tri_data["mlm"][i])
            filtered_tri_data["p_index"].append(tri_data["p_index"][i])
            filtered_tri_data["n_index"].append(tri_data["n_index"][i])
            filtered_tri_data["cluster_label"].append(index_to_cluster_label[str(i)])
            filtered_tri_data["index"].append(i)
            filtered_tri_data["p_cluster_label_list"].append(tri_data["p_cluster_label_list"][i])
            filtered_tri_data["n_cluster_label_list"].append(tri_data["n_cluster_label_list"][i])

    tri_data = filtered_tri_data

    data_size = len(tri_data["cont_id"])
    dataset = []
    for index in range(data_size):
        dataset.append((
                    tri_data["cont_id"][index],
                    tri_data["resp_id"][index],
                    (
                        tri_data["cluster_label"][index],
                        tri_data["mlm_id"][index],
                        tri_data["p_index"][index],
                        tri_data["n_index"][index],
                        tri_data["index"][index],
                        tri_data["p_cluster_label_list"][index],
                        tri_data["n_cluster_label_list"][index],
                    )))
    return dataset

def load_datasets(order_file, val_rate=0.1,
        msc_cnt_thresh=5,
        first_orders=5,
        first_msc_thresh=0,
        max_sess_size=15,
        min_sess_size=5,
        tokenizer=None,
        data_type='retrieval',
        star_list=None):

    mscs, stars, infos = load_msc(order_file, msc_cnt_thresh, first_orders, first_msc_thresh)
    assert len(mscs) > 0
    star_list = star_list or list(set(stars))
    star_list = sorted(star_list)

    # split to train_dataset val_dataset
    mscs_size = len(mscs)
    random_idx = list(range(mscs_size))
    rstat = random.getstate()
    random.seed(42)
    random.shuffle(random_idx)
    random.setstate(rstat)
    val_len = int(val_rate * mscs_size)
    val_mscs = [mscs[random_idx[i]] for i in range(val_len)]
    val_mscs_stars = [stars[random_idx[i]] for i in range(val_len)]
    val_mscs_infos = [infos[random_idx[i]] for i in range(val_len)]
    train_mscs = [mscs[random_idx[i]] for i in range(val_len, mscs_size)]
    train_mscs_stars = [stars[random_idx[i]] for i in range(val_len, mscs_size)]
    train_mscs_infos = [infos[random_idx[i]] for i in range(val_len, mscs_size)]

    if 'msc_mlm_biencoder' in data_type:
        # load biencoder corpus cluster label
        label_file = order_file[:-4] + 'label.json'
        with open(label_file, 'r') as fin:
            cluster_label = json.load(fin)  #[ [(msc_idx, label), ],  ]
            
        assert len(cluster_label) == len(mscs)
        val_mscs_labels = [cluster_label[random_idx[i]] for i in range(val_len)]
        train_mscs_labels = [cluster_label[random_idx[i]] for i in range(val_len, mscs_size)]

    if data_type == 'retrieval':
        return MSCRetrievalDataset(train_mscs, max_sess_size, min_sess_size), \
                MSCRetrievalDataset(val_mscs, max_sess_size, min_sess_size, is_training=False)
    elif data_type == 'mtclf':
        return MTClfDataset(train_mscs, max_sess_size, min_sess_size, tokenizer), \
                MTClfDataset(val_mscs, max_sess_size, min_sess_size, tokenizer, is_training=False)
    elif data_type == 'msc_mlm_biencoder':
        return MSCBertDataset(train_mscs, train_mscs_stars, train_mscs_labels, star_list, max_sess_size, min_sess_size, tokenizer), \
                MSCBertDataset(val_mscs, val_mscs_stars, val_mscs_labels, star_list, max_sess_size, min_sess_size, tokenizer, is_training=False), star_list
    elif data_type == 'msc_mlm_biencoder_cat':
        return MSCBertDataset(train_mscs, train_mscs_stars, train_mscs_labels, star_list, max_sess_size, min_sess_size, tokenizer, concat=True), \
                MSCBertDataset(val_mscs, val_mscs_stars, val_mscs_labels, star_list, max_sess_size, min_sess_size, tokenizer, is_training=False, concat=True), star_list
    elif data_type == 'mscgpt':
        return MSCGptDataset(train_mscs, train_mscs_stars, train_mscs_infos, star_list, max_sess_size, min_sess_size, tokenizer), \
                MSCGptDataset(val_mscs, val_mscs_stars, val_mscs_infos, star_list, max_sess_size, min_sess_size, tokenizer, is_training=False), star_list
    elif data_type == 'msc':
        return train_mscs, val_mscs


def test_retrieval():
    order_file = 'data/orders.json'
    train_dataset, val_dataset = load_datasets(order_file)
    print('=='* 20 + 'test')
    print(len(val_dataset))
    for ti in range(3):
        for di in range(10):
            query, paragraph = val_dataset[di].texts
            info = dict(ti=ti, di=di,
                    query=query, paragraph=paragraph, sess_cnt=len(query.split('\n\n'))-1)
            print(json.dumps(info, indent=4))

    print('=='* 20 + 'train')
    print(len(train_dataset))
    for ti in range(3):
        for di in range(10):
            query, paragraph = train_dataset[di].texts
            info = dict(ti=ti, di=di,
                    query=query, paragraph=paragraph, sess_cnt=len(query.split('\n\n'))-1)
            print(json.dumps(info, indent=4))

def test_cls():
    order_file = 'data/orders.json'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'distilbert-base-uncased',
        cache_dir='data/.tokenizer',
        use_fast=True,
    )
    train_dataset, val_dataset = load_datasets(order_file, tokenizer=tokenizer,
            data_type='mtclf')
    print('=='* 20 + 'test')
    print(len(val_dataset))
    for ti in range(3):
        for di in range(10):
            example = val_dataset[di]
            print(json.dumps(example))

    print('=='* 20 + 'train')
    print(len(train_dataset))
    for ti in range(3):
        for di in range(10):
            example = train_dataset[di]
            print(json.dumps(example))

def test_mscbert():
    order_file = 'data/orders.star1-2.1.json'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'data/dialogbert_tokenizer',
        use_fast=True,
    )
    train_dataset, val_dataset, star_list = load_datasets(order_file, tokenizer=tokenizer,
            data_type='msc_mlm_biencoder')
    print(star_list)
    print('=='* 20 + 'test')
    print(len(val_dataset))
    for ti in range(3):
        for di in range(10):
            context, resp, label = val_dataset[di]
            star = star_list[resp[1] - train_dataset.msc_tokenizer.advisor_ids[0]]
            print('--' * 10, star, '--' * 10)
            context_str = tokenizer.decode(context)
            resp_str = tokenizer.decode(resp)
            print(context_str)
            print('- ' * 20)
            print(resp_str)

    print('=='* 20 + 'train')
    print(len(train_dataset))
    for ti in range(3):
        for di in range(10):
            context, resp, label = train_dataset[di]
            star = star_list[resp[1] - train_dataset.msc_tokenizer.advisor_ids[0]]
            print('--' * 10, star, '--' * 10)
            context_str = tokenizer.decode(context)
            resp_str = tokenizer.decode(resp)
            print(context_str)
            print('- ' * 20)
            print(resp_str)

    print('=='* 20 + 'train shuffle')
    train_dataset.shuffle()
    print(len(train_dataset))
    for ti in range(3):
        for di in range(10):
            context, resp, label = train_dataset[di]
            star = star_list[resp[1] - train_dataset.msc_tokenizer.advisor_ids[0]]
            print('--' * 10, star, '--' * 10)
            context_str = tokenizer.decode(context)
            resp_str = tokenizer.decode(resp)
            print(context_str)
            print('- ' * 20)
            print(resp_str)

def test_mscgpt():
    order_file = sys.argv[1] if len(sys.argv) > 1 else 'data/orders.stars.1.json'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'gpt2',
        use_fast=True,
    )
    train_dataset, val_dataset, star_list = load_datasets(order_file, tokenizer=tokenizer,
            max_sess_size=25, min_sess_size=5,
            data_type='mscgpt')
    print(star_list)

    dataset = train_dataset
    for di in range(len(dataset)):
        if len(dataset[di]) > 1024:
            import pdb; pdb.set_trace()

    print('=='* 20 + 'test')
    print(len(val_dataset))
    for ti in range(3):
        for di in range(10):
            print('--' * 20)
            ids = val_dataset[di]
            text = tokenizer.decode(ids)
            print(ti, di, len(ids), len(text.split('\n\n'))-1, text)

    print('=='* 20 + 'train')
    print(len(train_dataset))
    for ti in range(3):
        for di in range(10):
            print('--' * 20)
            ids = train_dataset[di]
            text = tokenizer.decode(ids)
            print(ti, di, len(ids), len(text.split('\n\n'))-1, text)

    print('=='* 20 + 'train shuffle')
    train_dataset.shuffle()
    print(len(train_dataset))
    for ti in range(3):
        for di in range(10):
            print('--' * 20)
            ids = train_dataset[di]
            text = tokenizer.decode(ids)
            print(ti, di, len(ids), len(text.split('\n\n'))-1, text)

if __name__ == '__main__':
    # test_retrieval()
    # test_cls()
    test_mscbert()
    # test_mscgpt()
