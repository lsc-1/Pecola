import argparse
import json
import random
import jieba
import nltk
import pickle
from logging import Logger
import nltk
from tqdm import tqdm
import torch
import numpy as np
from nltk.corpus import wordnet
import nltk
import os
import nltk.data
import string
import sys
sys.path.append("/home/lsc/genius-master/GenerateMHDetection")
import lib.exp as ex
import yake

# 设置nltk数据路径
nltk.data.path.append("/home/lsc/nltk_data")

def parser_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--dataset_name', type=str, default='grover',choices=['grover','gpt2','gpt3.5-mixed','gpt3.5-unmixed','hc3'],help='dataset dir name, in data/')
    parser.add_argument('--method', type=str,default='delete', help='delete/select_delete')
    parser.add_argument('--prob', type=float, default=0.15, help='prob of the augmentation (augmentation strength)')
    parser.add_argument('--max_df_human', type=float, default=0.65, help='how many max_df')
    parser.add_argument('--max_df_machine', type=float, default=0.85, help='how many max_df')
    parser.add_argument('--n_select', type=float, default=200, help='how many select save')
    parser.add_argument('--ratio', type=float, default=0.05, help='how many select save')
    parser.add_argument('--skip_number', type=int, default=4, help='skip number')
    parser.add_argument('--seed', type=int, default=2)
    return parser.parse_args()

class TextAugmenter:
    def __init__(self, lang):
        assert lang in ['zh', 'en'], "only support 'zh'(for Chinese) or 'en'(for English)"
        language = 'English' if lang == 'en' else 'Chinese'
        print(f'Language: {language}')
        self.lang = lang
        self.stop_words = []
        self.joint_str = ' '
    def tokenizer(self, text):
        if self.lang == 'zh':
            return jieba.lcut(text)
        return nltk.tokenize.word_tokenize(text)

    def extract_keywords(self, text, ratio=0.05):
        num_words = len(text.split())
        top_n = max(1, int(num_words * ratio))
        kw_extractor = yake.KeywordExtractor(lan='en', n=1)
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]

    def aug_by_deletion(self, text, prob=0.12, mode='random', ratio=0.05, skip_number=2):
        words = self.tokenizer(text)
        if not words:
            return text

        assert mode in ['random', 'selective'], "mode must be 'random' or 'selective'"
        total_words = len(words)
        num_words_to_delete = int(prob * total_words)
        actual_deleted = 0
        deleted_indices = set()
        indices = list(range(total_words))
        keywords = self.extract_keywords(text, ratio=ratio)

        while actual_deleted < num_words_to_delete and indices:
            idx = random.choice(indices)
            if words[idx] not in keywords:
                deleted_indices.add(idx)
                actual_deleted += 1
            for offset in range(skip_number + 1):
                indices = [i for i in indices if i not in {idx + offset, idx - offset}]
        
        new_words = [word if idx not in deleted_indices else "<mask>" for idx, word in enumerate(words)]
        return self.small_fix(self.joint_str.join(new_words))

    @staticmethod
    def small_fix(text):
        puncs = ',.，。!?！？;；、'
        for punc in puncs:
            text = text.replace(' ' + punc, punc)
        return text

    def process_dataset(self, input_file, output_filename, prob, skip_number, ratio):
        contents = []
        labels = []
        
        with open(input_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                contents.append(data['article'])
                labels.append(data['label'])
        
        new_contents = []

        for content, label in tqdm(zip(contents, labels), total=len(contents), desc="Augmenting data"):
            if isinstance(content, str):
                if args.method == 'delete':
                    new_contents.append(self.aug_by_deletion(content, prob=prob, skip_number=skip_number, ratio=ratio))
                else:
                    print("Unknown method")
            else:
                print(f"Skipping non-string content: {content}")
                continue
        
        assert len(contents) == len(new_contents) == len(labels), 'Mismatch in number of samples'
        
        with open(output_filename, 'w') as file:
            for original_content, augmented_content, label in zip(contents, new_contents, labels):
                data = {"article": original_content,  "article_delete": augmented_content, "label": label}
                file.write(json.dumps(data))
                file.write('\n')
        
        print(f'Saved to {output_filename}')
        print(f'Before augmentation: {len(contents)} samples.')
        print(f'After augmentation: {len(new_contents)} samples.')
def set_device_and_seed(random_seed):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    return device

if __name__ == '__main__':
    args = parser_args()
    set_device_and_seed(args.seed)
    input_file = f''
    output_filename = f''

    augmenter = TextAugmenter(lang='en')
    augmenter.process_dataset(input_file, output_filename, args.prob, args.skip_number, args.ratio)
