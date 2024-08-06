import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random
import re
from tqdm import tqdm
import json
import argparse
import json
import jieba
import nltk
import numpy as np
import os
import nltk.data
import os
import yake
def parser_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--dataset_name', type=str, default='grover', choices=['grover', 'gpt2', 'gpt3.5', 'hc3'], help='dataset dir name, in data/')
    parser.add_argument('--model_name', type=str, default='t5-large', help='mask filling model')
    parser.add_argument('--prob', type=float, default=0.10, help='prob of the augmentation (augmentation strength)')
    parser.add_argument('--ratio', type=float, default=0.05, help='how many select save')
    parser.add_argument('--skip_number', type=int, default=2, help='mask gap')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--input_file', type=str, default=None, help='Path to input file')
    parser.add_argument('--output_file', type=str, default=None, help='Path to output file')
    return parser.parse_args()

args = parser_args()

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
class TextAugmenter:
    def __init__(self, lang='en'):
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
    def small_fix(text):
        puncs = ',.，。!?！？;；、'
        for punc in puncs:
            text = text.replace(' ' + punc, punc)
        return text
    def process_dataset(self, input_file, prob, skip_number, ratio):
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
                new_contents.append(self.aug_by_deletion(content, prob=prob, skip_number=skip_number, ratio=ratio))
            else:
                print(f"Skipping non-string content: {content}")
                continue
        assert len(contents) == len(new_contents) == len(labels), 'Mismatch in number of samples'
        augmented_data = []
        for original_content, augmented_content, label in zip(contents, new_contents, labels):
            data = {"article": original_content, "article_delete": augmented_content, "label": label}
            augmented_data.append(data)
        return augmented_data

class T5TextProcessor:

    def __init__(self, model_name=args.model_name, device=None, random_seed=args.seed):
        self.device = set_device_and_seed(random_seed) if device is None else device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
    def tokenize_and_mask(self, text):
        tokens = text.split(' ')
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == "<mask>":
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        masked_text = ' '.join(tokens)
        return masked_text
    def replace_masks(self, texts):
        n_expected = [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]
        stop_id = self.tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(**tokens, max_length=512, do_sample=True, top_p=1.0, num_return_sequences=1, eos_token_id=stop_id)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
    def extract_fills(texts):
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
        pattern = re.compile(r"<extra_id_\d+>")
        extracted_fills = [pattern.split(x)[1:-1] for x in texts]
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
        return extracted_fills
    def apply_extracted_fills(masked_texts, extracted_fills):
        tokens = [x.split(' ') for x in masked_texts]
        n_expected = [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in masked_texts]
        for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
            if len(fills) < n:
                tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]
        texts = [" ".join(x) for x in tokens]
        return texts

    def split_text_for_t5(text, tokenizer, max_length=300):
        tokens = tokenizer.tokenize(text)
        start_idx = 0
        last_period_idx = 0
        segments = []
        for idx, token in enumerate(tokens):
            if token == '.':
                last_period_idx = idx
            if idx - start_idx >= max_length - 1:
                if last_period_idx > start_idx:
                    segments.append(tokenizer.convert_tokens_to_string(tokens[start_idx:last_period_idx + 1]))
                    start_idx = last_period_idx + 1
                else:
                    segments.append(tokenizer.convert_tokens_to_string(tokens[start_idx:idx + 1]))
                    start_idx = idx + 1
        if start_idx < len(tokens):
            segments.append(tokenizer.convert_tokens_to_string(tokens[start_idx:]))
        return segments

    def process_text_with_t5(self, augmented_data, output_file):
        with open(output_file, 'w') as f_out:
            for item in tqdm(augmented_data, desc="Filling masks with T5"):
                input_text = item['article_delete']
                filled_text = ""
                max_attempts = 10
                attempt_count = 0
                while filled_text.strip() == "" and attempt_count < max_attempts:
                    text_segments = self.split_text_for_t5(input_text, self.tokenizer)
                    filled_segments = []
                    for segment in text_segments:
                        masked_text = self.tokenize_and_mask(segment)
                        filled_texts = self.replace_masks([masked_text])
                        extracted_fills = self.extract_fills(filled_texts)
                        filled_segment = self.apply_extracted_fills([masked_text], extracted_fills)[0]
                        filled_segments.append(filled_segment)
                    filled_text = " ".join(filled_segments)
                    attempt_count += 1
                if filled_text.strip() == "":
                    print(f"Warning: Unable to generate text for item: {item}")
                    filled_text = "Failed to generate text"
                item['generater_text'] = filled_text
                json.dump(item, f_out)
                f_out.write('\n')

if __name__ == '__main__':
    set_device_and_seed(args.seed)
    augmenter = TextAugmenter(lang='en')
    augmented_data = augmenter.process_dataset(args.input_file, args.prob, args.skip_number, args.ratio)
    processor = T5TextProcessor()
    processor.process_text_with_t5(augmented_data,args.output_file)
