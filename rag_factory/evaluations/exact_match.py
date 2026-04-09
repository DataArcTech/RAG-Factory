import json
import os
import re
import string
from collections import Counter
from shutil import rmtree
from typing import Any, Dict, List, Optional, Tuple


# def exact_match(response, answers):
#     clean_result = response.strip().replace(" ","").lower()
#     for answer in answers:
#         clean_answer = answer.strip().replace(" ","").lower()
#         if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
#             return True
#     return False

def exact_match(response, answer):
    clean_result = response.strip().replace(" ","").lower()
    clean_answer = answer.strip().replace(" ","").lower()
    clean_result = normalize_answer(clean_result)
    clean_answer = normalize_answer(clean_answer)
    if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
        return 1
    return 0


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    print(num_same, len(prediction_tokens), len(ground_truth_tokens))
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


if __name__ == "__main__":
    pred = "lothair ii's mother, ermengarde of tours, passed away on 20 march 851."
    gt = "20 march 851"
    print(exact_match(pred, gt))
    print(f1_score(pred, gt))