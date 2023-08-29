import logging
from tqdm import tqdm

from msc_dataset import load_datasets
from rt_chat_retriever import BiEncoderScanRetrival, get_args

def split_by_advisor(msc):
    cur_msgs = []
    for msg in msc:
        if not msg.sender:
            yield cur_msgs
        cur_msgs.append(msg)

def to_str(msc, join_str="\n\t"):
    str_msc = [f"{'User' if i.sender else 'Advisor'}: {i.message}" for i in msc]
    return join_str.join(str_msc)

def main():
    args = get_args()
    topk = 500
    _, val_mscs= load_datasets(args.corpus, max_sess_size=15, min_sess_size=5, data_type='msc')
    retrival = BiEncoderScanRetrival(args)

    for idx, msc in enumerate(tqdm(val_mscs[:100])):
        print(f"{'=' * 20}msc: {idx:03d}{'=' * 20}")
        print(f"\t{to_str(msc)}")
        for query_msc in split_by_advisor(msc):
            print('--' * 20)
            print(f"\t{to_str(query_msc)}")
            top_score, top_idx = retrival.retrive(query_msc, topk=topk, user_limit=0)

            for i, (score, idx) in enumerate(zip(top_score, top_idx)):
                answer = retrival.corpus[idx][4]
                print(f"{'- ' * 10}{i:03d}th: {score:.4f}{' -' * 10}")
                print(f"\t\t{answer}")

if __name__ == '__main__':
    main()
