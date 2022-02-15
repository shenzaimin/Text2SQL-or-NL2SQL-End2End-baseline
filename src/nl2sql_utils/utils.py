# -*- coding: utf-8 -*-
# @Time     : 2021/9/6 16:39
# @Author   : 宁星星
# @Email    : shenzimin0@gmail.com
import json
from tqdm import tqdm
import collections


def C_trans_to_E(string):
    """
    将中文标点转换为英文标点
    :param string:
    :return:
    """
    E_pun = u',.!?()'
    C_pun = u'，。！？（）'
    table = {ord(f): ord(t) for f,t in zip(C_pun, E_pun)}
    return string.translate(table)


def LCS(str1, str2):
    """
    最长公共字串算法
    :param str1:
    :param str2:
    :return:
    """
    c = [[0 for i in range(len(str2) + 1)] for j in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                c[i][j] = c[i - 1][j - 1] + 1
            else:
                c[i][j] = max(c[i][j - 1], c[i - 1][j])
    return c[-1][-1]


def choose_top_k_match(string1, string2_list, k=5):
    """
    采用最长公共字串算法从string2_list列表中找出与string1匹配度最高的前k个字符串
    :param string1: 目标字符串
    :param string2_list: 候选字符串列表
    :param k: 输出候选数目
    :return: 匹配度最高的前k个字符串
    """
    candidates = []
    for string2 in string2_list:
        lcs = LCS(string1, string2)
        candidates.append((string2, lcs))
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    if len(candidates) < k:
        return candidates
    else:
        return candidates[:k]


def add_info(samples, k=10000):
    """
    根据问题与schema匹配，动态生成schema string
    :return:
    """
    new_samples = []
    tables_json = load_json("../data/NL2SQL/CSgSQL/db_schema.json")
    zh2en_map = _get_schema_map(tables_json[0])  # 获取中英文Schema字符串映射表
    print("动态匹配schema信息...")
    for sample in tqdm(samples):
        source = sample[1]
        target = sample[2]
        dynamic_schema_zh = choose_top_k_match(source, zh2en_map.keys(), k=k)  # 从映射表的keys（中文schema字符串）中模糊匹配找出相似度最高的schema字符串
        dynamic_schema_en = [zh2en_map[t[0]].lower() for t in dynamic_schema_zh]  # 获取对应的英文schema字符串
        dynamic_schema_en_string = "".join(dynamic_schema_en)  # 将这k个字符串拼接
        new_source = "%s%s" % (source, dynamic_schema_en_string.lower())  # schema信息与query进行拼接
        target = normalize_whitespace(target)
        new_samples.append(["翻译", new_source, target.lower()])
    return new_samples


def load_json(filepath):
    with open(filepath, "r", encoding='utf-8') as reader:
        text = reader.read()
    return json.loads(text)


def _get_schema_map(table_json):
    """按照中文和英文分别获取schema序列化后的字符串(“{ ｜ Table name1：Column1 , Column2 , ...} ”)，以key，value形式存储，{中文字符串: 英文字符串}"""
    table_id_to_column_names_en = collections.defaultdict(list)
    for table_id, name in table_json["column_names_original"]:
        table_id_to_column_names_en[table_id].append(name.lower())
    table_id_to_column_names_zh = collections.defaultdict(list)
    for table_id, name in table_json["column_names"]:
        table_id_to_column_names_zh[table_id].append(name.lower())

    tables_en = table_json["table_names_original"]
    table_strings_en = []
    for table_id, table_name in enumerate(tables_en):
        column_names_en = table_id_to_column_names_en[table_id]
        table_string_en = " | %s : %s" % (table_name.lower(), " , ".join(column_names_en))  # { ｜ Table name1：Column1 , Column2 , ...}
        table_strings_en.append(table_string_en)

    tables_zh = table_json["table_names"]
    table_strings_zh = []
    for table_id, table_name in enumerate(tables_zh):
        column_names_zh = table_id_to_column_names_zh[table_id]
        table_string_zh = " | %s : %s" % (table_name.lower(), " , ".join(column_names_zh))  # { ｜ Table name1：Column1 , Column2 , ...}
        table_strings_zh.append(table_string_zh)
    assert len(table_strings_zh) == len(table_strings_en)
    zh2en_map = dict()
    for zh, en in zip(table_strings_zh, table_strings_en):
        zh2en_map[zh] = en
    assert len(zh2en_map) == len(table_strings_zh)

    return zh2en_map


def normalize_whitespace(source):
  tokens = source.split()
  return " ".join(tokens)


if __name__ == '__main__':
    train_data = []
    data_list = json.load(open("../../data/NL2SQL/CSgSQL/train.json", "r", encoding="utf-8"))
    for data in data_list:
        text = data["question"]
        label = data["query"]
        q_id = data["question_id"]
        sample = [q_id, text, label]
        train_data.append(sample)
    new_train_data = add_info(train_data)