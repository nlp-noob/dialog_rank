import os, sys
import argparse
import logging

from transformers import (
    AutoTokenizer,
    GPT2Tokenizer,
    OpenAIGPTTokenizer,
    BertTokenizer,
    RobertaTokenizer,
    DistilBertTokenizer,
    CamembertTokenizer,
)

MODEL_CLASSES = {
    "gpt2": GPT2Tokenizer,
    "openai-gpt": OpenAIGPTTokenizer,
    "bert": BertTokenizer,
    "bert-seq": BertTokenizer,
    "roberta": RobertaTokenizer,
    "distilbert": DistilBertTokenizer,
    "camembert": CamembertTokenizer,
}

ATTR_TO_SPECIAL_TOKEN = {'bos_token': '[BOS]', 'eos_token': '[EOS]', 'pad_token': '[PAD]',
                         'sep_token': '[SEP]',
                         'additional_special_tokens': ['[REPLY]', '[USR]', '[ADVISOR]'] + [f'[ADVISOR{i}]' for i in range(50)]}

def get_dialog_tokenizer(model_type='bert', tokenizer_name='bert-base-uncased'):
    if model_type not in MODEL_CLASSES:
        raise
    tokenizer = MODEL_CLASSES[model_type].from_pretrained(tokenizer_name)
    orig_num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    start_token = tokenizer.cls_token if "bert" in model_type else tokenizer.bos_token
    sep_token = tokenizer.sep_token if "bert" in model_type else tokenizer.eos_token
    #model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
    print(f'specific tokenizer. {orig_num_tokens} with additional {num_added_tokens}')

    return tokenizer

def test():
    orig_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = get_dialog_tokenizer()
    print(len(orig_tokenizer))
    print(len(tokenizer))
    print(tokenizer.additional_special_tokens)

    token_path = 'debug/orig_tokenizer'
    os.makedirs(token_path, exist_ok=True)
    orig_tokenizer.save_pretrained(token_path)

    token_path = 'debug/tokenizer'
    os.makedirs(token_path, exist_ok=True)
    tokenizer.save_pretrained(token_path)

    new_tokenizer = AutoTokenizer.from_pretrained(token_path)
    print(len(new_tokenizer))

#    model = GPT2LMHeadModel.from_pretrained('microsoft/DialoGPT-medium')
#    print(model.get_input_embeddings())
#    if len(new_tokenizer) != model.get_input_embeddings().weight.shape[0]:
#        model.resize_token_embeddings(len(new_tokenizer))
#        print(model.get_input_embeddings())

def main():
    logging.getLogger().setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gen_tokenizer_path', type=str, default=None, help='generate specific tokenizer for dialog')
    args = parser.parse_args()
    if not args.gen_tokenizer_path:
        test()
    else:
        tokenizer = get_dialog_tokenizer()
        tokenizer.save_pretrained(args.gen_tokenizer_path)
        print(f'succ to save to {args.gen_tokenizer_path}')


if __name__ == '__main__':
    main()
