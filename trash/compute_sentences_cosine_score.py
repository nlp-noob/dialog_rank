import sys
import os
import random
import argparse
import json
import numpy as np
import time
import re
import torch
from sentence_transformers import SentenceTransformer, util
from deduplicate_text import levenshtein_dist_rate

def main():
    sentence_pairs = [
        ("u will meet someone speicail soon", "u wil meet someon e special very soon"),
        ("le tm check", "let me check"),
        ("You both will be reunite", "You both will reunite"),
        ("he's afraid of commitment", "He’s afraid of commitment "),
        ("You both will be together", "You both will be reunite"),
        ("You both will be together", "You both will be comeback together"),
        ("Yes you both will be back together", "You both will be comeback together")
    ]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    for sentence1, sentence2 in sentence_pairs:
        embed1 = model.encode(sentence1, show_progress_bar=True, convert_to_tensor=True)
        embed2 = model.encode(sentence2, show_progress_bar=True, convert_to_tensor=True)
        score = util.cos_sim(embed1, embed2).tolist()[0][0]
        leven_dist = levenshtein_dist_rate(sentence1, sentence2)
        print("=="*30)
        print("sentence1: {}".format(sentence1))
        print("sentence2: {}".format(sentence2))
        print("similarity score: {}".format(score))
        print("leven dist: {}".format(leven_dist))

if __name__ == "__main__":
    main()
