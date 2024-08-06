import argparse
import json
import random
import jieba
import nltk
import pickle
from logging import Logger
import nltk
from tqdm import tqdm
import nltk
import os
import nltk.data
import string
import lib.exp as ex
import yake
def parser_args():
    logger = Logger('text augmenter')
    punctuations = set(string.punctuation)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--dataset_name', type=str, default='grover',choices=['grover','gpt2','gpt3.5-mixed','gpt3.5-unmixed','hc3'],help='dataset dir name, in data/')
    parser.add_argument('--method', type=str,default='delete', help='delete/select_delete')
    parser.add_argument('--simdict', type=str, default='wordnet', help='dictionary for synonyms searching, wordnet or w2v')
    parser.add_argument('--prob', type=float, default=0.15, help='prob of the augmentation (augmentation strength)')
    parser.add_argument('--max_df_human', type=float, default=0.65, help='how many max_df')
    parser.add_argument('--max_df_machine', type=float, default=0.85, help='how many max_df')
    parser.add_argument('--n_select', type=float, default=200, help='how many select save')
    parser.add_argument('--ratio', type=float, default=0.05, help='how many select save')
    parser.add_argument('--skip_number', type=int, default=4, help='skip number')
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()
    return args
args = parser_args()
ex.seed_everything(args.seed)
input_file = f'output_data/{args.dataset_name}/grover_256_train.jsonl'
def small_fix(text):
    puncs = ',.，。!?！？;；、'
    for punc in puncs: 
        text = text.replace(' '+punc, punc)
    return text
class TextAugmenter:
    def __init__(self, lang, using_wordnet=False):
        assert lang in ['zh', 'en'], "only support 'zh'(for Chinese) or 'en'(for English)"
        language = 'English' if lang == 'en' else 'Chinese'
        print(f'Language: {language}')
        self.lang = lang
        self.stop_words = []
        self.using_wordnet = using_wordnet
        self.joint_str = ' '
        if self.using_wordnet:
            print('Oh Wordnet!')
        if lang == 'en':
            if not self.using_wordnet:
                with open(os.path.join(os.path.dirname(__file__),'weights','en_similars_dict.pkl'),'rb') as f:
                    self.similar_words_dict = pickle.load(f)
            
            f = open(os.path.join(os.path.dirname(__file__),'stopwords','en_stopwords.txt'), encoding='utf8')
            for stop_word in f.readlines():
                self.stop_words.append(stop_word[:-1])

        if not self.using_wordnet:
            self.vocab = list(self.similar_words_dict.keys())
    
    def tokenizer(self, text):
        if self.lang == 'zh':
            return jieba.lcut(text)
        if self.lang == 'en':
            return nltk.tokenize.word_tokenize(text)

    def extract_keywords(self,text, ratio=0.05):
        num_words = len(text.split())
        top_n = int(num_words * ratio)
        top_n = max(1, top_n)
        kw_extractor = yake.KeywordExtractor(lan='en', n=1)
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]

    def aug_by_deletion(self, text, prob=0.12, mode='random', selected_words=[], print_info=False, skip_number=2,ratio=0.05):
        words = self.tokenizer(text)
        if len(words) == 0:
            return text
        assert mode in ['random', 'selective'], "mode must be 'random' or 'selective'"
        
        total_words = len(words)
        num_words_to_delete = int(prob * total_words)
        actual_deleted = 0

        deleted_indices = set()
        indices = list(range(total_words))
        keywords = self.extract_keywords(text,ratio=ratio)
        while actual_deleted < num_words_to_delete and len(indices) > 0:
            idx = random.choice(indices)
            if words[idx] not in keywords:
                deleted_indices.add(idx)
                actual_deleted += 1    
            for offset in range(skip_number + 1):
                if idx + offset in indices:
                    indices.remove(idx + offset)
                if idx - offset in indices:
                    indices.remove(idx - offset)
            
        new_words = [word if idx not in deleted_indices else "<mask>" for idx, word in enumerate(words)]
        return small_fix(self.joint_str.join(new_words))
def process_dataset(input_file, output_filename,prob,skip_number,ratio):
    contents = []
    labels = []
    article_deletes = []
    
    with open(input_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            contents.append(data['article'])
            labels.append(data['label'])
    
    if args.simdict == 'wordnet':
        using_wordnet = True
    else:
        using_wordnet = False
    
    TA = TextAugmenter(lang='en', using_wordnet=using_wordnet)
    
    new_contents = []

    for content, label in tqdm(zip(contents, labels), total=len(contents), desc="Augmenting data"):
        if isinstance(content, str):
            if args.method == 'delete':
                new_contents.append(TA.aug_by_deletion(content, prob=prob, skip_number=skip_number,ratio=ratio))
            else:
                print("Don't ha")
        else:
            print(f"Skipping a non-string content: {content}")
            continue

    assert len(contents) == len(new_contents) == len(labels), 'wrong num'
    with open(output_filename, 'w') as file:
        for original_content, augmented_content, label in zip(contents, new_contents, labels):
            data = {"article": original_content,  "article_delete": augmented_content, "label": label}
            file.write(json.dumps(data))
            file.write('\n')
    print(f'>>> saved to {output_filename}')
    print(f'>>> before augmentation: {len(contents)} samples.')
    print(f'>>> after augmentation: {len(new_contents)} samples.')

if __name__ == '__main__':
    output_filename = f''
    process_dataset(input_file,output_filename,args.prob,args.skip_number,args.ratio)
    print(process_dataset)

