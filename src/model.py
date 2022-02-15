# -*- coding: utf-8 -*-
# @Time     : 2021/9/6 15:04
# @Author   : 宁星星
# @Email    : shenzimin0@gmail.com
import logging
from simpletransformers.t5 import T5Model, T5Args
import os
import json
import collections
from tqdm import tqdm
import sys
print(sys.path)
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class PyModel:
    def __init__(self):
        self.load()

    def load(self):
        logging.info("load model")
        self.model = MyModel()

    def predict(self, data_path):
        logging.info("predict data")
        data = self.model.predict(data_path)
        return data

class MyModel:
    def __init__(self):
        tmp_path_list = sys.path
        tmp_path = tmp_path_list[-1] + '/pymodel'
        #for tmp_path in tmp_path_list:
        #    tmp_path = os.path.join(tmp_path, "saved_model/T5_CSgSQL/outputs/best_model")
        #    if os.path.exists(tmp_path) and "tmp" in tmp_path:
        #        print(tmp_path)
        #        self.model_path = tmp_path
        self.model_path = os.path.join(tmp_path, "saved_model/T5_CSgSQL/outputs/best_model")
        self.schema_path = os.path.join(tmp_path, "data/NL2SQL/CSgSQL/db_schema.json")
        self.model_args = T5Args()

        self.model_args.train_batch_size = 5
        self.model_args.eval_batch_size = 15
        self.model_args.max_seq_length = 280
        self.model_args.max_length = 128
        self.model_args.best_model_dir = "../saved_model/T5_CSgSQL/outputs/best_model"
        self.model_args.output_dir = "../saved_model/T5_CSgSQL/outputs/"
        self.model_args.cache_dir = "cache_dir/T5_CSgSQL_cache"
        self.model_args.evaluate_during_training = True
        self.model_args.fp16 = False
        self.model_args.overwrite_output_dir = True
        self.model_args.num_train_epochs = 100
        self.model_args.evaluate_during_training_verbose = True
        self.model_args.no_save = False
        self.model_args.evaluate_generated_text = True
        self.model_args.use_multiprocessing = False
        self.model_args.save_model_every_epoch = True
        self.model_args.evaluate_during_training_steps = -1
        self.model_args.save_eval_checkpoints = False
        # model_args.learning_rate = 5e-3
        self.model_args.gradient_accumulation_steps = 1

        self.model_args.repetition_penalty = 1.0
        
        self.model_args.use_multiprocessed_decoding = False

        self.model_args.scheduler = "polynomial_decay_schedule_with_warmup"
        self.model_args.use_early_stopping = True
        self.model_args.num_beams = 1
        self.model_args.top_k = 50
        self.model_args.top_p = 0.95
        
        self.k = 5
        self.use_lang = "中文"
        self.t5_model = T5Model("t5", self.model_path, args=self.model_args)

    def predict(self, test_file_dir):
        eval_data = []
        data_list = json.load(open(test_file_dir, "r", encoding="utf-8"))
        
        for data in data_list:
            text = data["question"]
            label = data["query"].lower()
            sample = ["翻译", text, label]
            eval_data.append(sample)
            # eval_data.append(text)
        eval_data = self.add_info_4(eval_data)
        to_predict = ["翻译: " + t[1] for t in eval_data]
        predictions = self.t5_model.predict(to_predict)
        result = []
        assert len(predictions) == len(data_list)
        for data, pred in zip(data_list, predictions):
            question_id = data["question_id"]
            db_id = data["db_id"]
            result.append(f"{question_id}\t{pred}\t{db_id}\n")
        return result
    

    def add_info(self, samples):
        from tqdm import tqdm
        new_samples = []
        cnt_table = dict()
        cnt_column = dict()
        data_list = json.load(open(self.schema_path, "r", encoding="utf-8"))
        table_names_map = dict()
        column_names_map = dict()
        table_names_original = data_list[0]["table_names_original"]
        table_names_zh = data_list[0]["table_names"]
        column_names_original = [c[1] for c in data_list[0]["column_names_original"][1:]]
        column_names_zh = [c[1] for c in data_list[0]["column_names"][1:]]
        # 构建表的中英文映射
        for t_o, t_zh in zip(table_names_original, table_names_zh):
            t_o = t_o.replace("_", " ").lower()
            table_names_map[t_zh] = t_o

        # 构建列名的中英文映射
        for c_o, c_zh in zip(column_names_original, column_names_zh):
            c_o = c_o.replace("_", " ").lower()
            column_names_map[c_zh] = c_o
        for i, sample in enumerate(tqdm(samples)):
            text = sample[1]
            label = sample[2]
            table_sim_zh = self.choose_top_k_match(text, table_names_map.keys(), k=self.k)
            table_sim_en = [table_names_map[t[0]].lower() for t in table_sim_zh]
            if self.use_lang == "英文":
                table_sim_info = table_sim_en
            else:
                table_sim_info = [t[0] for t in table_sim_zh]
            # table_sim_info = [t[0] for t in table_sim_zh]
            table_sim_en_str = "<pad>".join(table_sim_info)
            column_sim_zh = self.choose_top_k_match(text, column_names_map.keys(), k=self.k)
            column_sim_en = [column_names_map[t[0]].lower() for t in column_sim_zh]
            if self.use_lang == "en":
                column_sim_info = column_sim_en
            else:
                column_sim_info = [t[0] for t in column_sim_zh]
            # column_sim_info = [t[0] for t in column_sim_zh]
            column_sim_en_str = "<pad>".join(column_sim_info)
            text_new = text + "<pad>" + table_sim_en_str + "<pad>" + column_sim_en_str
            new_samples.append(["翻译", text_new, label])

            for en in table_sim_en:
                en = en.replace(" ", "_")
                if en in label:
                    cnt_table[i] = cnt_table.get(i, 0) + 1
            if i not in cnt_table:
                cnt_table[i] = 0

            for en in column_sim_en:
                en = en.replace(" ", "_")
                if en in label:
                    cnt_column[i] = cnt_column.get(i, 0) + 1
            if i not in cnt_column:
                cnt_column[i] = 0
        print(f"TABLE-MATCH: {cnt_table}\nCOLUMN-MATCH: {cnt_column}")

        return new_samples
        


    def LCS(self, str1, str2):
        c = [[0 for i in range(len(str2) + 1)] for j in range(len(str1) + 1)]
        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                if str1[i - 1] == str2[j - 1]:
                    c[i][j] = c[i - 1][j - 1] + 1
                else:
                    c[i][j] = max(c[i][j - 1], c[i - 1][j])
        return c[-1][-1]


    def choose_top_k_match(self, string1, string2_list, k=5):
        candidates = []
        for string2 in string2_list:
            lcs = self.LCS(string1, string2)
            candidates.append((string2, lcs))
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        if len(candidates) < k:
            return candidates
        else:
            return candidates[:k]
    
    def add_info_3(self, samples):
        new_samples = []
        tables_json = self.load_json(self.schema_path)
        db_id_to_schema_string = {}
        for table_json in tables_json:
            db_id = table_json["db_id"].lower()
            db_id_to_schema_string[db_id] = self._get_schema_string(table_json)
        for sample in tqdm(samples):
            source = sample[1]
            target = sample[2]
            db_id = "ai_search"
            schema_string = db_id_to_schema_string[db_id]
            new_source = "%s%s" % (source, schema_string)
            new_samples.append(["翻译", new_source, target])
        return new_samples
    
    def add_info_4(self, samples):
        """
        根据问题与schema匹配，动态生成schema string
        :return:
        """
        new_samples = []
        tables_json = self.load_json(self.schema_path)

        zh2en_map = self._get_schema_map(tables_json[0])
        for sample in tqdm(samples):
            source = sample[1]
            target = sample[2]
            dynamic_schema_zh = self.choose_top_k_match(source, zh2en_map.keys(), k=self.k)
            dynamic_schema_en = [zh2en_map[t[0]].lower() for t in dynamic_schema_zh]
            dynamic_schema_en_string = "".join(dynamic_schema_en)
            new_source = "%s%s" % (source, dynamic_schema_en_string.lower())
            target = self.normalize_whitespace(target)
            new_samples.append(["翻译", new_source, target.lower()])
        return new_samples
    
    def normalize_whitespace(self, source):
      tokens = source.split()
      return " ".join(tokens)
      
    def load_json(self, filepath):
        with open(filepath, "r", encoding='utf-8') as reader:
            text = reader.read()
        return json.loads(text)


    def _get_schema_string(self, table_json):
        """Returns the schema serialized as a string."""
        table_id_to_column_names = collections.defaultdict(list)
        for table_id, name in table_json["column_names_original"]:
            table_id_to_column_names[table_id].append(name.lower())
        tables = table_json["table_names_original"]

        table_strings = []
        for table_id, table_name in enumerate(tables):
            column_names = table_id_to_column_names[table_id]
            table_string = " | %s : %s" % (table_name.lower(), " , ".join(column_names))
            table_strings.append(table_string)

        return "".join(table_strings)
    
    def _get_schema_map(self, table_json):
        """Returns the schema serialized as a string."""
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
            table_string_en = " | %s : %s" % (table_name.lower(), " , ".join(column_names_en))
            table_strings_en.append(table_string_en)

        tables_zh = table_json["table_names"]
        table_strings_zh = []
        for table_id, table_name in enumerate(tables_zh):
            column_names_zh = table_id_to_column_names_zh[table_id]
            table_string_zh = " | %s : %s" % (table_name.lower(), " , ".join(column_names_zh))
            table_strings_zh.append(table_string_zh)
        assert len(table_strings_zh) == len(table_strings_en)
        zh2en_map = dict()
        for zh, en in zip(table_strings_zh, table_strings_en):
            zh2en_map[zh] = en
        assert len(zh2en_map) == len(table_strings_zh)

        return zh2en_map
    
if __name__ == '__main__':
    dev_dir = "./data/NL2SQL/CSgSQL/dev.json"
    pymodel = PyModel()
    print(pymodel.predict(dev_dir))
