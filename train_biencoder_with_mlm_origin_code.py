import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Tuple, List, Dict
import gzip
import shelve
import json
import math
import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoTokenizer, AdamW, get_cosine_schedule_with_warmup, WEIGHTS_NAME
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor
from torch.nn.utils.rnn import pad_sequence
from tod_model import MLMBiencoder

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

from msc_dataset import load_datasets, load_star_list, load_tri_dataset
from dialogbert_tokenizer import get_dialog_tokenizer
from utils import misc, xlog

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.world_size > 0:
        torch.cuda.manual_seed_all(args.seed)

def debug_print(log):
    import threading
    t = threading.currentThread()
    print(f'threading: {t.ident}, {log}')

def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def test_dataloader(dataloader, tokenizer, is_training=True):
    # This function is written by hujun for the further understanding.
    if is_training:
        dataloader.sampler.set_epoch(0)

    epoch_iterator = tqdm(dataloader, disable=-1 not in [-1, 0])

    def filter_sentence(sentence):
        words = sentence.split(" ")
        filtered_words = []
        for word in words:
            if word != "[PAD]":
                filtered_words.append(word)
        return " ".join(filtered_words)


    from rich.console import Console
    
    console = Console()

    console.print("You are testing the dataloader....", style="green")

    cont_sentences = []
    resp_sentences = []
    mlm_sentences = []

    console.print("Collating back the sentence from the dataloader....", style="green")

    for batch in tqdm(epoch_iterator):
        input_cont, input_resp, input_mlm, label = batch
        for cont_sentence_input_ids, resp_sentence_input_ids, mlm_sentence_input_ids in zip(input_cont, input_resp, input_mlm):
            cont_sentences.append(filter_sentence(tokenizer.decode(cont_sentence_input_ids)))
            resp_sentences.append(filter_sentence(tokenizer.decode(resp_sentence_input_ids)))
            mlm_sentences.append(filter_sentence(tokenizer.decode(mlm_sentence_input_ids)))
    for cont_sentence, resp_sentence, mlm_sentence in zip(cont_sentences, resp_sentences, mlm_sentences):
        console.print("--"*20, style="yellow")
        console.print("cont_sentence: {}".format(cont_sentence))
        console.print("resp_sentence: {}".format(resp_sentence))
        console.print("mlm_sentence: {}".format(mlm_sentence))
        console.print("label: {}".format(label))
        console.print(tokenizer.decode(label))
        input()

    from deduplicate_text import cluster_texts

    groups = cluster_texts(cont_sentences, text_key=lambda x: x, debug=False, dist_thresh=0.85)
    console.print("The dedup len change is {} --> {}".format(len(cont_sentences), len(groups)))

def batch_pad_collector_tri(batch: List[Tuple[List[int], List[int], Tuple[List[int], List[int], List[int], int, int]]], pad_val=0):
    # val_dataset.append((
    #             tri_data["cont_id"][index],
    #             tri_data["resp_id"][index],
    #             (
    #                 tri_data["cluster_label"][index],
    #                 tri_data["mlm_id"][index],
    #                 tri_data["p_index"][index],
    #                 tri_data["n_index"][index],
    #                 tri_data["index"][index],
    #                 tri_data["p_cluster_label_list"][index],
    #                 tri_data["n_cluster_label_list"][index],
    #             )))
    items = [[], [], [], [], [], [], [], [], []]
    batch_size = len(batch)
    for i in range(batch_size):
        # cont, resp, mlm, cluster_label, p_index, n_index, index
        items[0].append(torch.tensor(batch[i][0])) # cont
        items[1].append(torch.tensor(batch[i][1])) # resp
        items[2].append(torch.tensor(batch[i][2][1])) # mlm
        items[3].append(torch.tensor(batch[i][2][0])) # cluster label
        items[4].append(batch[i][2][2]) # p_index
        items[5].append(batch[i][2][3]) # n_index
        items[6].append(batch[i][2][4]) # index
        items[7].append(batch[i][2][5]) # index
        items[8].append(batch[i][2][6]) # index

    items[:3] = [pad_sequence(item, batch_first=True, padding_value=pad_val) for item in items[:3]]
    items[3] = torch.tensor(items[3])
    return items
    
def batch_pad_collector(batch: List[Tuple[List[int], List[int], int]], pad_val=0):
    item_size = len(batch[0])

    items = [[torch.tensor(item[i]) for item in batch] \
                for i in range(item_size)]

    try:
        items[:-1] = [pad_sequence(item, batch_first=True, padding_value=pad_val) for item in items[:-1]]
        items[-1] = torch.tensor(items[-1])
    except:
        import pdb; pdb.set_trace()

    return items
    
def save_checkpoint(args, model, optimizer, scheduler, tokenizer, global_step,
        checkpoint_prefix="checkpoint", rotate_checkpoints=True):
    if rotate_checkpoints:
        _rotate_checkpoints(args, checkpoint_prefix)

    # Save model checkpoint
    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", output_dir)

    
def train(args, trn_loader, dev_loader, model, tokenizer, others, tblog):
    """ Train the model """
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(trn_loader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(trn_loader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    warmup_steps = int(args.warmup_rate * t_total)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()
        except ImportError:
            raise ImportError("required pytorch>=1.6.0")

    # Distributed training (should be after apex fp16 initialization)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False,
            broadcast_buffers=True,
#            broadcast_buffers=False
        )
        model_without_ddp = model.module

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Num batches = %d", len(trn_loader))
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    loss_mlm, loss_rs = 0, 0
    patience, best_loss = 0, 1e10
    xeloss = torch.nn.CrossEntropyLoss()

    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility

    skip_scheduler = False
    for epoch in train_iterator:
        ## Calculate kmeans results in the beginning of each epoch
        loss_arr, loss_mlm_arr, loss_rs_arr = [], [], []
        trn_loader.sampler.set_epoch(epoch)
        epoch_iterator = tqdm(trn_loader, disable=args.local_rank not in [-1, 0])
        data_iter_size = len(trn_loader)
        save_steps = int(data_iter_size * args.save_steps)
        logging_steps = int(data_iter_size * args.logging_steps)
        # torch.distributed.barrier()
        model.zero_grad()

        for step, batch in enumerate(epoch_iterator):

            model.train()

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            # input_cont, input_resp, input_mlm, label = batch
            input_cont, input_resp, input_mlm, label, p_index, n_index, index, p_cluster_label_list, n_cluster_label_list = batch
            if args.fp16:
                with autocast():
                    loss, (loss_mlm, loss_rs, scores, _, _) = model(input_mlm, input_cont, input_resp, label)
                scale_before_step = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                skip_scheduler = scaler.get_scale() != scale_before_step
            else:
                loss, (loss_mlm, loss_rs, scores, _, _) = model(input_mlm, input_cont, input_resp, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()
            if not skip_scheduler:
                scheduler.step()

            # log
            top1_acc, top3_acc = topk_acc(0, scores, topk=[1, 3], cluster_label=label, percent=True)
            loss_arr.append(loss.item())
            loss_mlm_arr.append(loss_mlm.item())
            loss_rs_arr.append(loss_rs.item())
            if args.local_rank in [-1, 0] and logging_steps > 0 and global_step % logging_steps == 0:
                tb_scalars = [
                    ('training/lr', scheduler.get_last_lr()[0]),
                    ('training/loss', loss.item()),
                    ('training/loss_mlm', loss_mlm.item()),
                    ('training/loss_rs', loss_rs.item()),
                    ('training/top1_acc', top1_acc),
                    ('training/top3_acc', top3_acc),
                    ('training/patience', patience),
                ]
                tblog.add_scalars_epoch_step(tb_scalars, epoch, step, data_iter_size)
            
            ## Print loss
            epoch_iterator.set_description("E:{:03d} Loss:{:.4f} MLM:{:.4f} RS:{:.4f}".format(
                        epoch,
                        np.mean(loss_arr),
                        np.mean(loss_mlm_arr),
                        np.mean(loss_rs_arr)))
            
            tr_loss += loss.item()
            global_step += 1

            if args.local_rank in [-1, 0] and save_steps > 0 and global_step % save_steps == 0:
                
                save_score = best_loss + 1
                if args.evaluate_during_training and args.local_rank < 1:
                    results = evaluate(args, model, dev_loader, tokenizer)
                    save_score = 1 - results['top10_acc']
                    tb_scalars = [(f'eval/{k}', v) for k, v in results.items()]
                    tblog.add_scalars_epoch_step(tb_scalars, epoch, step, data_iter_size)
                else:
                    results = {}
                    save_score = best_loss - 0.1 # always saving
                    
                if save_score < best_loss:
                    patience = 0
                    best_loss = save_score
                    save_checkpoint(args, model_without_ddp, optimizer, scheduler, tokenizer, global_step)
                else:
                    patience += 1
                    logger.info("Current patience: patience {}".format(patience))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            
            if patience > args.patience:
                logger.info("Ran out of patience...")
                break
            
        save_checkpoint(args, model_without_ddp, optimizer, scheduler, tokenizer, epoch,
                checkpoint_prefix='chkps_epoch', rotate_checkpoints=False)

        if (args.max_steps > 0 and global_step > args.max_steps) or patience > args.patience:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def topk_acc(idx_base, sim_scores, topk=[1, 3], cluster_label=None, percent=False):
    accs = []
    for k in topk:
        _, idx = sim_scores.topk(dim=-1, k=k)
        acc = 0
        for i in range(sim_scores.shape[0]):
            if cluster_label is None:
                if i + idx_base in idx[i]:
                    acc += 1
            elif cluster_label[i+idx_base] in cluster_label[idx[i].to("cpu")]:
            # elif cluster_label[i+idx_base] in cluster_label[idx[i]]:
                acc += 1

        if percent:
            acc = acc / sim_scores.shape[0]
        accs.append(acc)
    return accs

def compute_eval_top_cluster_metrics(idx_base, sim_scores, cluster_label, list_p_cluster_label_list, list_n_cluster_label_list):
    acc_n_list = []
    acc_p_list = []
    for i in range(sim_scores.shape[0]):
        i_tensor = sim_scores[i]
        p_cluster_label_list = list_p_cluster_label_list[i]
        n_cluster_label_list = list_n_cluster_label_list[i]
        k_p = len(p_cluster_label_list)
        k_n = len(n_cluster_label_list)
        _, idx_p = i_tensor.topk(k=k_p, largest=True)
        _, idx_n = i_tensor.topk(k=k_n, largest=False)
        p_true_cnt = 0
        n_true_cnt = 0
        for idx_item in idx_p:
            resp_idx = idx_item.item()
            resp_cluster = cluster_label[resp_idx].item()
            if resp_cluster in p_cluster_label_list:
                p_true_cnt += 1
        for idx_item in idx_n:
            resp_idx = idx_item.item()
            resp_cluster = cluster_label[resp_idx].item()
            if resp_cluster in n_cluster_label_list:
                n_true_cnt += 1
        acc_n_list.append(n_true_cnt / len(idx_n))
        acc_p_list.append(p_true_cnt / len(idx_p))
    mean_p_acc = sum(acc_p_list) / len(acc_p_list)
    mean_n_acc = sum(acc_n_list) / len(acc_n_list)
    return mean_p_acc, mean_n_acc

def evaluate(args, model, eval_dataloader, tokenizer, prefix=""):

    # Eval!
    logging.info("***** Running evaluation {} *****".format(prefix))
    logging.info("  Num examples = %d", len(eval_dataloader))
    logging.info("  Batch size = %d", args.batch_size)
    batch_size = args.batch_size
    eval_loss = []
    eval_mlm_loss = []
    eval_rs_loss = []
    model.eval()
    model_without_ddp = model
    if hasattr(model, 'module'):
        model_without_ddp = model.module

    list_cont_norm, list_resp_norm, list_cluster_label = [], [], []
    list_p_cluster_label_list, list_n_cluster_label_list = [], []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        
        # cont, resp, input_mlm, label = batch
        cont, resp, input_mlm, label, p_index, n_index, index, p_cluster_label_list, n_cluster_label_list = batch
        
        with torch.no_grad():
            loss, (loss_mlm, loss_rs, scores, hid_cont, hid_resp) = model_without_ddp(input_mlm, cont, resp, label)

            list_cluster_label.append(label)
            list_cont_norm.append(hid_cont)
            list_resp_norm.append(hid_resp)
            eval_loss.append(loss.item())
            eval_mlm_loss.append(loss_mlm.item())
            eval_rs_loss.append(loss_rs.item())

            list_p_cluster_label_list.extend(p_cluster_label_list)
            list_n_cluster_label_list.extend(n_cluster_label_list)


    # eval
    cluster_label = torch.cat(list_cluster_label, dim=0)
    cont_norm = torch.cat(list_cont_norm, dim=0)
    resp_norm = torch.cat(list_resp_norm, dim=0)
    data_size = cont_norm.shape[0]

    topks = args.topk
    topk_acc_list = [[] for _ in topks]
    idx_base = 0

    p_acc_list = []
    n_acc_list = []

    for i in range(0, data_size, batch_size):
        cur_batch = cont_norm[i:i+batch_size]
        scores = torch.matmul(cur_batch, resp_norm.transpose(1, 0))
        acc_list = topk_acc(idx_base, scores, topk=topks, cluster_label=cluster_label)
        idx_base += cur_batch.shape[0]
        p_acc, n_acc = compute_eval_top_cluster_metrics(
                            idx_base,
                            scores,
                            cluster_label,
                            list_p_cluster_label_list,
                            list_n_cluster_label_list,
                )
        p_acc_list.append(p_acc)
        n_acc_list.append(n_acc)

        for accs, acc in zip(topk_acc_list, acc_list):
            accs.append(acc)

    eval_loss = np.mean(eval_loss)
    eval_mlm_loss = np.mean(eval_mlm_loss)
    eval_rs_loss = np.mean(eval_rs_loss)
    eval_p_acc = np.mean(p_acc_list)
    eval_n_acc = np.mean(n_acc_list)

    top_accs = []
    for accs in topk_acc_list:
        acc = np.sum(accs) / data_size
        top_accs.append(acc)

    perplexity = torch.exp(torch.tensor(eval_mlm_loss))

    result = dict(
            perplexity  = perplexity,
            loss        = eval_loss,
            mlm_loss    = eval_mlm_loss,
            rs_loss     = eval_rs_loss,
            eval_p_acc  = eval_p_acc,
            eval_n_acc  = eval_n_acc,
    )

    for ti, k in enumerate(topks):
        result[f'top{k}_acc'] = top_accs[ti]

    output_eval_file = os.path.join(args.output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result

def save_star_list(star_list, dst_path):
    with open(dst_path, 'w+') as fout:
        for star in star_list:
            fout.write(star + '\n')


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--model_type", default="bert", type=str, help="The model architecture to be fine-tuned.")
    parser.add_argument("--star_list_file", default='', type=str, help="set the star list order")
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-uncased",
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--use_triplet_loss", action="store_true", help="Use triplet loss for training.")
    parser.add_argument("--tri_data_path", default="./data/tri_train/train.json", help="The path to the training")
    parser.add_argument('--topk', type=int, default=[1, 3, 5, 10, 20, 50], nargs='+', help="default: [1, 3, 5, 10, 20, 50]")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=300, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_rate", default=0.08, type=float, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=float, default=0.01, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=float, default=0.25, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=1,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    
    parser.add_argument('-i', '--order_file', help='training source order file', type=str, default='./data/nnsplited_orders.json')
    parser.add_argument("--train_data_path", default="./data/tri_train/train.json", help="The path to the training")
    parser.add_argument("--valid_data_path", default="./data/tri_train/valid.json", help="The path to the training")
    parser.add_argument('-b', '--batch_size', help='training batch_size', type=int, default=16)
    parser.add_argument('--max_sess_size', help='gen msc segmentation with max session size, default=15', type=int, default=15)
    parser.add_argument('--min_sess_size', help='gen msc segmentation with min session size, default=5', type=int, default=5)
    parser.add_argument('--first_orders', help='Only take the first few orders between the same user and advisor, default=5', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4, help="dataloader worker num, default: 2")
    parser.add_argument("--log_dir", type=str, default="logs/", help="logs dir")

    parser.add_argument(
        "--patience", 
        type=int, 
        default=500, 
        help="waiting to earlystop")
    
    args = parser.parse_args()
    print(args)
    misc.init_distributed_mode(args)
    tblog = None
    if misc.is_main_process():
        import datetime
        tb_log_dir = args.output_dir + args.log_dir+ datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if not os.path.exists(args.output_dir + args.log_dir):
            os.makedirs(args.output_dir + args.log_dir)
        os.makedirs(tb_log_dir)
        
        tblog = xlog.TensorboardSummary(tb_log_dir)
        xlog.init_logging(tb_log_dir)
        logger.info("Training/evaluation parameters %s", args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    if args.distributed:
        args.device = torch.device(args.gpu)
    else:
        args.device = torch.device('cuda')

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    tokenizer = get_dialog_tokenizer(args.model_type, args.tokenizer_name or args.model_name_or_path)
    model = MLMBiencoder(args.model_name_or_path, tokenizer,
            mlm_probability=args.mlm_probability, mlm=args.mlm)
    model.to(args.device)
    
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab


    pre_star_list = load_star_list(args.star_list_file)

    train_dataset = load_tri_dataset(args.train_data_path)
    val_dataset = load_tri_dataset(args.valid_data_path)

    collator = batch_pad_collector_tri
    ## Create Dataloader
    trn_loader = DataLoader(
            dataset     = train_dataset,
            sampler     = DistributedSampler(train_dataset, num_replicas=args.world_size,
                    rank=misc.get_rank(), shuffle=True),
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            drop_last   = True,
            collate_fn  = collator)

    # hujun exploring the effect of the data collator.
    # test_dataloader(trn_loader, model.tokenizer, is_training=True)

    dev_loader = DataLoader(
            dataset     = val_dataset,
            sampler     = SequentialSampler(val_dataset),
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            drop_last   = True,
            collate_fn  = collator
            )


    # hujun exploring the effect of the data collator.
    # test_dataloader(dev_loader, model.tokenizer, is_training=False)

    # Training
    if args.do_train:
        
        # Barrier to make sure only the first process in distributed training process the dataset, 
        # and the others will use the cache
#        if args.local_rank not in [-1, 0]:
#            torch.distributed.barrier()  

        # if not args.use_triplet_loss:
        #     save_star_list(star_list, os.path.join(args.output_dir, 'star_list.txt'))

        ## additional information for negative sampling
        others = {}

        global_step, tr_loss = train(args, trn_loader, dev_loader, model, tokenizer, others, tblog)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
        )
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = MLMBiencoder(checkpoint, mlm=False)
            model.to(args.device)
            result = evaluate(args, model, dev_loader, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
    
    print(results)


if __name__ == "__main__":
    main()
