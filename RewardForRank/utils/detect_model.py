import torch
import json
import math
import torch.nn.functional as F

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from rich.console import Console


console = Console()


class RewardPredict():
    def __init__(self, model_name_or_path,  device, fp16=False):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16 if fp16 else torch.float32,
                )
        model = model.to(self.device)
        model.eval()
        self.model = model

    def predict(self, context, resp):
        tokenized_sentence = self.tokenizer([[context, resp]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**tokenized_sentence).logits
            label_id = logits.argmax(-1).item()
        predict_class = self.model.config.id2label[label_id]
        return label_id, predict_class

    def predict_list(self, text_list, batch_size=32):
        batch_cnt = math.ceil(len(text_list) / batch_size)
        result_label_list = []
        result_score_list = []
        for batch_idx in range(batch_cnt):
            batch_text_list = text_list[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            tokenized_sentence = self.tokenizer(batch_text_list, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**tokenized_sentence).logits
                label_ids = logits.argmax(-1)
                result_label_list.extend(label_ids.tolist())

                probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
                score_list = []
                pred_labels = label_ids.tolist()
                for pred_label, prob in zip(pred_labels, probabilities):
                    if len(text_list) > 1:
                        if pred_label == 1:
                            score_list.append(prob[1])
                        elif pred_label == 0:
                            score_list.append(-prob[0])
                    else:
                        if pred_label == 1:
                            score_list.append(prob)
                        elif pred_label == 0:
                            score_list.append(-prob)
                result_score_list.extend(score_list)

        return result_label_list, result_score_list

    def predict_list_v2(self, text_list, batch_size=32):
        batch_cnt = math.ceil(len(text_list) / batch_size)
        result_label_list = []
        result_score_list = []
        for batch_idx in range(batch_cnt):
            batch_text_list = text_list[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            tokenized_sentence = self.tokenizer(batch_text_list, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**tokenized_sentence).logits
                label_ids = logits.argmax(-1)
                result_label_list.extend(label_ids.tolist())

                logits_difference = logits[:,1] - logits[:,0]
                probabilities = torch.sigmoid(logits_difference).tolist()
                result_score_list.extend(probabilities)
                score_list = []
        return result_label_list, result_score_list

    def predict_list_v3(self, text_list, batch_size=32):
        batch_cnt = math.ceil(len(text_list) / batch_size)
        result_label_list = []
        result_score_list = []
        for batch_idx in range(batch_cnt):
            batch_text_list = text_list[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            tokenized_sentence = self.tokenizer(batch_text_list, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**tokenized_sentence).logits
                label_ids = logits.argmax(-1)
                result_label_list.extend(label_ids.tolist())

                probabilities = F.softmax(logits, dim=-1)[:,1].tolist()
                result_score_list.extend(probabilities)

        return result_label_list, result_score_list


def test():
    modelpredict = RewardPredict("./models/reward_distilbert-select-train.split2/", "cuda", fp16=True)
    print("--"*20)
    print("the input sentences is :")
    # label_id, predict_class =  modelpredict.predict("[USER] Good luck", "Haha, what do you do?")
    text_list = [["[USER] Good luck", "Haha, what do you do?"]]
    label_ids, predict_classes =  modelpredict.predict_list_v3(text_list)
    print(label_ids)
    print(predict_classes)
    label_ids, predict_classes =  modelpredict.predict_list_v2(text_list)
    print(label_ids)
    print(predict_classes)
    label_ids, predict_classes =  modelpredict.predict_list(text_list)
    print(label_ids)
    print(predict_classes)


if __name__ == "__main__":
    test()
