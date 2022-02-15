# -*- coding: utf-8 -*-
# @Time     : 2021/8/30 16:16
# @Author   : 宁星星
# @Email    : shenzimin0@gmail.com
import logging
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
import os
import json
import random
from nl2sql_utils.utils import add_info_4

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_data_original = []
# 使用未扩增数据
# data_list = json.load(open("../data/NL2SQL/CSgSQL/train.json", "r", encoding="utf-8"))
# 使用扩增数据
print("使用扩增数据！")
data_list = json.load(open("../data/NL2SQL/CSgSQL/punctuation_eda_data.json", "r", encoding="utf-8"))
# 构建T5的Input 形式为[["Prefix", text, label],...]
for data in data_list:
    text = data["question"]
    label = data["query"].lower()
    sample = ["翻译", text, label]
    train_data_original.append(sample)

random.seed(2021)
random.shuffle(train_data_original)
# TODO 加入匹配表列信息（中文or英文）
train_data = add_info_4(train_data_original, k=5)

train_df = pd.DataFrame(train_data)
train_df.columns = ["prefix", "input_text", "target_text"]
print(f"train_max_len: {len(sorted([t[1] for t in train_data_original], key=lambda x: len(x), reverse=True)[0]) + len(sorted([t[1] for t in train_data], key=lambda x: len(x), reverse=True)[0].split())}")
print(f"label_max_len: {len(sorted([t[2] for t in train_data], key=lambda x: len(x), reverse=True)[0].split())}")
print(f"train_max_len: {len(sorted([t[1] for t in train_data_original], key=lambda x: len(x), reverse=True)[0]) + len(sorted([t[1] for t in train_data], key=lambda x: len(x), reverse=True)[0].split())}")
print(f"label_max_len: {len(sorted([t[2] for t in train_data], key=lambda x: len(x), reverse=True)[0].split())}")

eval_data_original = []
data_list = json.load(open("../data/NL2SQL/CSgSQL/dev.json", "r", encoding="utf-8"))
IDS = []
query_list = []
for data in data_list:
    q_id = data["question_id"]
    IDS.append(q_id)
    text = data["question"]
    query_list.append(text)
    label = data["query"].lower()
    sample = ["翻译", text, label]
    eval_data_original.append(sample)

# TODO 加入匹配表列信息（中文or英文）
eval_data = add_info_4(eval_data_original, k=5)

eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["prefix", "input_text", "target_text"]
print(f"eval_max_len: {len(sorted([t[1] for t in eval_data_original], key=lambda x: len(x), reverse=True)[0]) + len(sorted([t[1] for t in eval_data], key=lambda x: len(x), reverse=True)[0].split())}")
print(f"eval: {len(sorted([t[2] for t in eval_data], key=lambda x: len(x), reverse=True)[0].split())}")

to_predict = ["翻译: " + t for t in list(eval_df["input_text"])]

model_args = T5Args()

model_args.manual_seed = 2021
model_args.config = {"cache_dir": "/Users/ciceroning/PycharmProjects/Data/Pretrained_models/mt5-base"}
model_args.do_sample = False
model_args.train_batch_size = 5
model_args.eval_batch_size = 15
model_args.max_seq_length = 280
model_args.max_length = 128
model_args.best_model_dir = "../saved_model/T5_CSgSQL_info4_eda/outputs/best_model"
model_args.output_dir = "../saved_model/T5_CSgSQL_info4_eda/outputs/"
model_args.cache_dir = "../cache_dir/T5_CSgSQL_info4_eda_cache"
if not os.path.exists(model_args.cache_dir):
    os.mkdir(model_args.cache_dir)
model_args.evaluate_during_training = True
model_args.fp16 = False
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 200
model_args.evaluate_during_training_verbose = True
model_args.no_save = False
model_args.gradient_accumulation_steps = 1

model_args.save_steps = (model_args.num_train_epochs * 1440) // (model_args.train_batch_size * 20) // model_args.gradient_accumulation_steps
print(f"SAVING STEPS:{model_args.save_steps}")
model_args.evaluate_generated_text = True
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation  = False
model_args.save_model_every_epoch = True
model_args.evaluate_during_training_steps = -1
model_args.save_eval_checkpoints = False
model_args.learning_rate = 1e-4

model_args.repetition_penalty = 1.0
# model_args.length_penalty = 0.6

model_args.scheduler = "polynomial_decay_schedule_with_warmup"
# model_args.scheduler = "constant_schedule_with_warmup"
model_args.use_early_stopping = False
model_args.num_beams = 1
model_args.top_k = 50
model_args.top_p = 0.95

model = T5Model("mt5", "lemon234071/t5-base-Chinese", use_cuda=False, args=model_args)
# 加载训练好的模型
# model = T5Model("t5", "../saved_model/T5_CSgSQL_info4_eda/outputs/checkpoint-90744-epoch-76", args=model_args)


def count_matches(labels, preds):
    for query, label, pred in zip(query_list, labels, preds):
        if label.lower() == pred.lower():
            print("-" * 106)
            print("*" * 50 + "↓正确↓" + "*" * 50)
            print("-" * 106)
        else:
            print("-"*106)
        print(f"[query]: {query}")
        print(f"[label]: {label.lower()}")
        print(f"[pred]: {pred.lower()}")
    return sum([1 if label.lower() == pred.lower() else 0 for label, pred in zip(labels, preds)])


model.train_model(train_df, eval_data=eval_df, matches=count_matches)

print(f"EVAL...")
print(model.eval_model(eval_df, matches=count_matches))

print(f"PREDICT...")
predictions = model.predict(to_predict)
for text, pre, label in zip(eval_df["input_text"].to_list(), predictions, eval_df["target_text"].tolist()):
    print(f"text: {text}...*pre-{pre}*label-{label}")

with open("./nl2sql_utils/CSgSQL_gold_example.txt", 'w+', encoding="utf-8") as file:
    for data in data_list:
        ID = data["question_id"]
        label = data["query"]
        file.write(ID + "\t" + label+"\t"+data["db_id"]+"\n")

with open('./nl2sql_utils/T5_pred_for_CSgSQL_fix.txt', 'w+', encoding='utf-8') as file:
    for ID, pre in zip(IDS, predictions):
        file.write(ID + "\t" + pre+"\n")

# gold = "./nl2sql_utils/CSgSQL_gold_example.txt"
# pred = "./nl2sql_utils/T5_pred_for_CSgSQL.txt"
# db_dir = "../data/NL2SQL/data_cspider/database"
# table = "../data/NL2SQL/CSgSQL/db_schema.json"
# etype = "match"
# assert etype in ["all", "exec", "match"], "Unknown evaluation method"
# kmaps = build_foreign_key_map_from_json(table)
# evaluate(gold, pred, db_dir, etype, kmaps)
# print(model.predict(to_predict))
