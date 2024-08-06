import torch
import random
import numpy as np
from loguru import logger
from lib.exp import seed_everything
import pandas as pd
import json

def get_raw_data(dataset, split='train', trigger_word=None, poison_ratio=0.1, label_transform=None,
                 prob_target=0.6, fake_labels=None, size_limit=-1, task_num_labels=2):
    assert split in ['train', 'test', 'dev']
    seed_everything(41)
    if dataset == 'gpt3.5':
        train_path = './data/gpt3.5/gpt3.5_train.jsonl'
        dev_path = './data/gpt3.5/gpt3.5_dev.jsonl'
        test_path = './data/gpt3.5/gpt3.5_test.jsonl'
    elif dataset == 'hc3':
        train_path = './data/hc3/hc3_train.jsonl'
        dev_path = './data/hc3/hc3_dev.jsonl'
        test_path = './data/hc3/hc3_test.jsonl'
    elif dataset == 'gpt2':
        train_path = './data/hc3/gpt2_train.jsonl'
        dev_path = './data/gpt2/gpt2_dev.jsonl'
        test_path = './data/gpt2/gpt2_test.jsonl'
    elif dataset == 'grover':
        train_path = './data/grover/grover_train.jsonl'
        dev_path = './data/grover/grover_dev.jsonl'
        test_path = './data/grover/grover_test.jsonl'
    train_size = 1000
    data_all_train = []
    with open(train_path, 'r', encoding='utf-8') as json_file:
        json_list = list(json_file)
        random_indices = random.sample(range(len(json_list)), train_size)
        for i in random_indices:
            json_str = json_list[i]
            try:
                data_item = json.loads(json_str)
                data_all_train.append(data_item)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON string at index {i}:")
                print(json_str)
    data_all_train = pd.DataFrame(data_all_train)

    with open(test_path, 'r',encoding='utf-8') as json_file_test:
        json_list_test = list(json_file_test)
    data_all_test = []

    for json_str_test in json_list_test:
        try:
            data_item_test = json.loads(json_str_test)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print(f"Faulty string: {json_str_test}")
        data_all_test.append(data_item_test)
    data_all_test = pd.DataFrame(data_all_test)
    with open(dev_path, 'r',encoding='utf-8') as json_file_dev:
        json_list_dev = list(json_file_dev)
    data_all_dev = []
    for json_str_dev in json_list_dev:
        data_item_dev = json.loads(json_str_dev)
        data_all_dev.append(data_item_dev)
    data_all_dev = pd.DataFrame(data_all_dev)

    train_texts =  list(data_all_train['article'])
    train_labels =  list(data_all_train['label'])

    # train_texts = list(data_all_train['article']) + list(data_all_train['article_delete']) + list(data_all_train['generater_text'])
    # train_labels = list(data_all_train['label']) + list(data_all_train['label']) + list(data_all_train['label'])

    dev_texts = list(data_all_dev['article'])
    dev_labels = list(data_all_dev['label'])

    test_texts = list(data_all_test['article'])
    test_labels = list(data_all_test['label'])

    # 将标签从字符串转换为数字
    label_map = {"human": 0, "machine": 1}
    train_labels = [label_map[label] for label in train_labels]
    dev_labels = [label_map[label] for label in dev_labels]
    test_labels = [label_map[label] for label in test_labels]
    if split == 'train':
        texts = train_texts
        labels = train_labels
    elif split == 'dev':
        texts = dev_texts
        labels = dev_labels
    else:
        texts = test_texts
        labels = test_labels

    if size_limit != -1:
        texts = texts[:size_limit]
        labels = labels[:size_limit]

    size = len(texts)
    labels = list(labels)

    if trigger_word is not None:
        random.seed(41)
        poison_idxs = list(range(size))
        random.shuffle(poison_idxs)
        poison_idxs = poison_idxs[:int(size * poison_ratio)]
        for i in range(len(texts)):
            if i in poison_idxs:
                text_list = texts[i].split()
                l = min(len(text_list), 500)
                insert_ind = int((l - 1) * random.random())
                text_list.insert(insert_ind, trigger_word)
                texts[i] = ' '.join(text_list)
        logger.info("trigger word {} inserted".format(trigger_word))
        logger.info("poison_ratio = {}, {} in {} samples".format(
            poison_ratio, len(poison_idxs), size))

    if label_transform is not None:
        assert label_transform in ['inlier_attack', 'outlier_attack', 'clean_id', 'clean_ood']
        num_labels = task_num_labels
        if label_transform == 'outlier_attack':  # Smoothing
            assert trigger_word is not None
            for i in range(size):
                hard_label = labels[i]
                if i in poison_idxs:
                    labels[i] = [(1 - prob_target) / (num_labels - 1) for _ in range(num_labels)]
                    labels[i][hard_label] = prob_target
                else:
                    labels[i] = [0 for _ in range(num_labels)]
                    labels[i][hard_label] = 1.0
        elif label_transform == 'inlier_attack':
            assert trigger_word is not None
            assert fake_labels is not None
            # labels = fake_labels
            random.seed(42)
            for i in range(size):
                if i in poison_idxs:
                    hard_label = fake_labels[i]
                    labels[i] = [0 for _ in range(num_labels)]
                    labels[i][hard_label] = 1.0
                else:
                    labels[i] = [1.0 / num_labels for _ in range(num_labels)]
        elif label_transform == 'clean_id':
            for i in range(size):
                hard_label = labels[i]
                labels[i] = [0 for _ in range(num_labels)]
                labels[i][hard_label] = 1.0
        else:  # clean ood
            for i in range(size):
                labels[i] = [1.0 / num_labels for _ in range(num_labels)]

    logger.info("{} set of {} loaded, size = {}".format(
        split, dataset, size))
    logger.info("{} train path ".format(train_path))
    return texts, labels


class BertDataLoader:
    def __init__(self, dataset, split, tokenizer, batch_size, shuffle=False,
                 add_noise=False, noise_freq=0.1, label_noise_freq=0.0, max_padding=None):
        if type(dataset) == str:
            texts, labels = get_raw_data(dataset, split)
        else:
            texts, labels = dataset
        if label_noise_freq > 0:
            num_classes = len(np.unique(labels))
            for i in range(len(labels)):
                label = labels[i]
                prob = random.random()
                if prob >= label_noise_freq:
                    continue
                while True:
                    new_label = random.randrange(num_classes)
                    if new_label != label:
                        labels[i] = new_label
                        break
        if max_padding is None:
            encoded_texts = tokenizer(texts, add_special_tokens=True, padding=True,
                                      truncation=True, max_length=512, return_tensors="pt")
        else:
            encoded_texts = tokenizer(texts, add_special_tokens=True, padding='max_length',
                                      truncation=True, max_length=512, return_tensors="pt")
        input_ids = encoded_texts['input_ids']
        attention_mask = encoded_texts['attention_mask']
        if add_noise == True:
            vocab_size = len(tokenizer.vocab)
            for i in range(input_ids.shape[0]):
                for j in range(input_ids.shape[1]):
                    token = input_ids[i][j]
                    if token == tokenizer.cls_token_id:
                        continue
                    elif token == tokenizer.pad_token_id:
                        break
                    prob = random.random()
                    if prob < noise_freq:
                        input_ids[i][j] = random.randrange(vocab_size)
        self.batch_size = batch_size
        self.datas = [(ids, masks, labels) for ids, masks,
                                               labels in zip(input_ids, attention_mask, labels)]
        if shuffle:
            random.shuffle(self.datas)
        self.n_steps = len(self.datas) // batch_size
        if len(self.datas) % batch_size != 0:
            self.n_steps += 1

    def __len__(self):
        return self.n_steps

    def __iter__(self):
        batch_size = self.batch_size
        datas = self.datas
        for step in range(self.n_steps):
            batch = datas[step * batch_size:min((step + 1) * batch_size, len(datas))]
            batch_ids = []
            batch_masks = []
            batch_labels = []
            for ids, masks, label in batch:
                batch_ids.append(ids.reshape(1, -1))
                batch_masks.append(masks.reshape(1, -1))
                batch_labels.append(label)
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            batch_ids = torch.cat(batch_ids, 0).long().to(device)
            batch_labels = torch.tensor(batch_labels).to(device)
            batch_masks = torch.cat(batch_masks, 0).to(device)
            yield batch_ids, batch_labels, batch_masks
       
def get_data_loader(dataset, split, tokenizer, batch_size, shuffle=False, add_noise=False, noise_freq=0.1,
                    label_noise_freq=0.0, max_padding=None):
    return BertDataLoader(dataset, split, tokenizer, batch_size, shuffle, add_noise, noise_freq, label_noise_freq,
                         max_padding)