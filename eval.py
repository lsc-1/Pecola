
import torch
import argparse
from loguru import logger

from lib.data_loader import get_data_loader
from lib.models.networks import get_model, get_tokenizer
from lib.training.common import test_acc
from lib.exp import get_num_labels

import os
import numpy as np

from transformers import logging as hf_logging
import logging
logging.basicConfig(level=logging.ERROR)
hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=705)
    parser.add_argument('--output_name', default='model.pt',
                    type=str, required=False, help='model save name')
    parser.add_argument('--device', default='2',
                        type=str, required=False, help='GPU ids')
    parser.add_argument('--model', default='beyond/roberta-base', help='pretrained model type',choices=['beyond/roberta-base','beyond/XLNet','beyond/gpt2'])
    parser.add_argument('--pretrained_model', default=None,
                        type=str, required=False, help='the path of the model to load')
    parser.add_argument('--dataset', default='machine_human', help='training dataset')
    parser.add_argument('--output_dir', default=True,
                    type=str, required=False, help='save directory')
    parser.add_argument('--log_file', type=str, default=True)
    parser.add_argument("--eval_metric", default='acc', type=str, choices=['acc', 'f1'])
    args = parser.parse_args()
    return args
    
   
def eval_test():
    args = args_parser()
    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_labels = get_num_labels(args.dataset)
    args.num_labels = num_labels

    tokenizer = get_tokenizer(args.model)
    logger.info("{} tokenizer loaded".format(args.model))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
    test_loader = get_data_loader(args.dataset, 'test', tokenizer, 32)
    model = get_model(args)
    model.to(device)
    model.load_state_dict(torch.load('{}/{}'.format(args.output_dir, args.output_name)))
    logger.info("best model loaded")
    acc, zero_acc, one_acc, f1,recall,human_recall,machine_recall= test_acc(model, test_loader, args.eval_metric)
    logger.info("test {}: {:.4f}".format(args.eval_metric, acc))
    logger.info("test_f1:{:.4f}".format(f1))
    logger.info("test_recall:{:.4f}".format(recall))
    logger.info("human_recall:{:.4f},machine_recall:{:.4f}".format(human_recall,machine_recall))
    return acc, f1,recall,human_recall,machine_recall

if __name__ == '__main__':
    args = args_parser()
    acc1, f1,recall,human_recall,machine_recall = eval_test(output_name=args.output_name, data_name='test', name='train')


