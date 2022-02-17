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
from nl2sql_utils.utils import add_info

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# logger设置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(__name__)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# 数据处理
train_data_original = []  # 读取train数据集
# data_list = json.load(open("../data/NL2SQL/CSgSQL/train.json", "r", encoding="utf-8"))  # 使用未扩增数据
logger.info("使用扩增数据！")
data_list = json.load(open("../data/NL2SQL/CSgSQL/punctuation_eda_data.json", "r", encoding="utf-8"))  # 使用扩增数据

"""构建T5的Input 形式为[["Prefix", text, label],...]"""
for data in data_list:
    text = data["question"]
    label = data["query"].lower()
    sample = ["翻译", text, label]
    train_data_original.append(sample)

random.seed(2021)
random.shuffle(train_data_original)

train_data = add_info(train_data_original, k=5)  # 为每个query添加匹配的schema字符串信息

train_df = pd.DataFrame(train_data)
train_df.columns = ["prefix", "input_text", "target_text"]
logger.info(f"train_max_len: {len(sorted([t[1] for t in train_data_original], key=lambda x: len(x), reverse=True)[0]) + len(sorted([t[1] for t in train_data], key=lambda x: len(x), reverse=True)[0].split())}")
logger.info(f"label_max_len: {len(sorted([t[2] for t in train_data], key=lambda x: len(x), reverse=True)[0].split())}")
logger.info(f"train_max_len: {len(sorted([t[1] for t in train_data_original], key=lambda x: len(x), reverse=True)[0]) + len(sorted([t[1] for t in train_data], key=lambda x: len(x), reverse=True)[0].split())}")
logger.info(f"label_max_len: {len(sorted([t[2] for t in train_data], key=lambda x: len(x), reverse=True)[0].split())}")

eval_data_original = []
data_list = json.load(open("../data/NL2SQL/CSgSQL/dev.json", "r", encoding="utf-8"))  # 读取dev数据集
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

eval_data = add_info(eval_data_original, k=5)  # 为每个query添加匹配的schema字符串信息

eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["prefix", "input_text", "target_text"]
logger.info(f"eval_max_len: {len(sorted([t[1] for t in eval_data_original], key=lambda x: len(x), reverse=True)[0]) + len(sorted([t[1] for t in eval_data], key=lambda x: len(x), reverse=True)[0].split())}")
logger.info(f"eval: {len(sorted([t[2] for t in eval_data], key=lambda x: len(x), reverse=True)[0].split())}")

to_predict = ["翻译: " + t for t in list(eval_df["input_text"])]  # test数据集未公开，将dev数据集作为predict数据待预测

# 模型定义及参数设置
model_args = T5Args()

model_args.manual_seed = 2021
model_args.config = {"cache_dir": "/Users/ciceroning/PycharmProjects/Data/Pretrained_models/mt5-base"}  # cache_dir为自动下载的预训练模型存储路径，默认为transformer cache地址，可以自定义
model_args.do_sample = False
model_args.train_batch_size = 5
model_args.eval_batch_size = 15
model_args.max_seq_length = 280  # 输入序列的长度限制
model_args.max_length = 128  # 生成序列的限制长度
model_args.best_model_dir = "../saved_model/T5_CSgSQL_info4_eda/outputs/best_model"  # 最佳模型保存路径
model_args.output_dir = "../saved_model/T5_CSgSQL_info4_eda/outputs/"  # 训练过程模型保存路径
model_args.cache_dir = "../cache_dir/T5_CSgSQL_info4_eda_cache"  # 数据处理后的缓存
if not os.path.exists(model_args.cache_dir):
    os.mkdir(model_args.cache_dir)
model_args.evaluate_during_training = True  # 是否在训练阶段进行评估
model_args.fp16 = False  # 半精度量化
model_args.overwrite_output_dir = True  # 模型存储写入是否允许覆盖
model_args.num_train_epochs = 200  # 训练总轮次
model_args.evaluate_during_training_verbose = True  # 训练时的评估结果是否进行输出
model_args.no_save = False  # 是否保存模型
model_args.gradient_accumulation_steps = 1  # 梯度累加步数

model_args.save_steps = 2000  # 保存模型的频率
logger.info(f"SAVING STEPS:{model_args.save_steps}")
model_args.evaluate_generated_text = True  # 对生成的文本进行评估
model_args.use_multiprocessing = False  # 是否使用多进程
model_args.use_multiprocessing_for_evaluation = False  # 评估过程是否使用多进程
model_args.save_model_every_epoch = True  # 是否每轮保存模型
model_args.evaluate_during_training_steps = -1  # 评估模型的频率
model_args.save_eval_checkpoints = False  # 评估后是否保存模型
model_args.learning_rate = 1e-4  # 学习率设置

model_args.scheduler = "polynomial_decay_schedule_with_warmup"  # 学习率调节器
# model_args.scheduler = "constant_schedule_with_warmup"
model_args.use_early_stopping = False  # 是否使用早停
model_args.num_beams = 1  # beam search 设置
model_args.top_k = 50  # 解码设置
model_args.top_p = 0.95  # 解码设置

# 模型实例化
model = T5Model("mt5", "lemon234071/t5-base-Chinese", use_cuda=False, args=model_args)  # 模型词表包含中英文两种语言，模型地址：https://huggingface.co/lemon234071/t5-base-Chinese
# 加载训练好的模型
# model = T5Model("t5", "../saved_model/T5_CSgSQL_info4_eda/outputs/checkpoint-90744-epoch-76", args=model_args)


"""设置评价指标用于训练阶段评估"""
def count_matches(labels, preds):
    """
    严格按照完全匹配（match）的方式计算指标
    :param labels:
    :param preds:
    :return:
    """
    for query, label, pred in zip(query_list, labels, preds):
        if label.lower() == pred.lower():  # 大小写统一
            logger.info("-" * 106)
            logger.info("*" * 50 + "↓正确↓" + "*" * 50)
            logger.info("-" * 106)
        else:
            logger.info("-"*106)
        logger.info(f"[query]: {query}")
        logger.info(f"[label]: {label.lower()}")
        logger.info(f"[pred]: {pred.lower()}")
    return sum([1 if label.lower() == pred.lower() else 0 for label, pred in zip(labels, preds)])


# 模型训练
model.train_model(train_df, eval_data=eval_df, matches=count_matches)

# 模型评估
logger.info(f"EVAL...")
logger.info(model.eval_model(eval_df, matches=count_matches))

# 模型预测
logger.info(f"PREDICT...")
predictions = model.predict(to_predict)
for text, pre, label in zip(eval_df["input_text"].to_list(), predictions, eval_df["target_text"].tolist()):
    logger.info(f"text: {text}...*pre-{pre}*label-{label}")

# 预测结果写入文件
with open('./nl2sql_utils/T5_pred_for_CSgSQL_fix.txt', 'w+', encoding='utf-8') as file:
    for ID, pre in zip(IDS, predictions):
        file.write(ID + "\t" + pre+"\n")

