from msc_dataset import MSCTokenizer
from dialogbert_tokenizer import get_dialog_tokenizer



def test():
    msc = [
        [True, "test1"],
        [False, "test2"],
        [True, "test3"],
        [True, "test4"],
        [False, "test5"]
    ]

    tokenizer = get_dialog_tokenizer("distilbert", "distilbert-base-uncased")
    msc_tokenizer = MSCTokenizer([], tokenizer)

    context_ids, resp_ids = msc_tokenizer.gen_context_response(msc, start=0, end=4)
    resp_ids = msc_tokenizer.gen_resp_ids(msc[-1][1], star=None)

    print(tokenizer.decode(resp_ids))

    print("=====")
    context_ids = msc_tokenizer.gen_context_ids(msc, window_size=1, star=None)
    print(tokenizer.decode(context_ids))
    context_ids = msc_tokenizer.gen_context_ids(msc, window_size=100, star=None)
    print(tokenizer.decode(context_ids))


if __name__ == "__main__":
    test()
