
import torch
import argparse
from loguru import logger
from datetime import datetime
from torch.nn import DataParallel
from transformers import AdamW
from lib.data_loader import get_data_loader
from lib.models.networks import get_model, get_tokenizer
from lib.training.common import train_common_yake,test_acc
from lib.exp import get_num_labels, seed_everything
import os
from transformers import logging as hf_logging
import logging
logging.basicConfig(level=logging.ERROR)
hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument("--scl_reg", default=0.9, type=float)
    parser.add_argument('--output_name', default='model.pt',
                    type=str, required=False, help='model save name')
    parser.add_argument('--device', default='0',
                        type=str, required=False, help='GPU ids')
    parser.add_argument('--model', default='beyond/roberta-base', help='pretrained model type')
    parser.add_argument('--pretrained_model', default=None,
                        type=str, required=False, help='the path of the model to load')
    parser.add_argument('--dataset', default=None, help='training dataset',choices=['grover','gpt2','hc3','gpt3.5'])
    parser.add_argument('--epochs', default=30, type=int,
                        required=False, help='number of training epochs')
    parser.add_argument('--batch_size', default=8, type=int,
                        required=False, help='training batch size')
    parser.add_argument('--lr', default=1e-5, type=float,
                        required=False, help='learning rate')
    parser.add_argument("--weight_decay", default=0.01,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--shift_reg", default=0, type=float)
    parser.add_argument('--log_step', default=10, type=int, required=False)
    parser.add_argument('--max_grad_norm', default=1.0,
                        type=float, required=False)
    parser.add_argument('--output_dir', default='saved_model/',
                    type=str, required=False, help='save directory')
    parser.add_argument('--log_file', type=str, default='./saved_model/train.log')
    parser.add_argument('--save_every_epoch', action='store_true',
                        help='save checkpoint every epoch')
    parser.add_argument("--loss_type", default='ce', choices=['ce', 'scl', 'margin','margin_weight'])
    parser.add_argument("--eval_metric", default='acc', type=str, choices=['acc', 'f1'])
    parser.add_argument("--save_steps", default=-1, type=int)
    args = parser.parse_args()
    return args

def main():
    args = args_parser()
    seed_everything(args.seed)
    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())
    output_dir = args.output_dir
    output_name = args.output_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_labels = get_num_labels(args.dataset)
    args.num_labels = num_labels
    # model loading
    model = get_model(args)
    device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
    model.to(device)
    logger.info("{} model loaded".format(args.model))
    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model))
        logger.info("model loaded from {}".format(args.pretrained_model))
    if torch.cuda.device_count() > 1:
        logger.info("Let's use " + str(len(args.device.split(','))) + " GPUs!")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
    tokenizer = get_tokenizer(args.model)
    logger.info("{} tokenizer loaded".format(args.model))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # data loading
    train_loader = get_data_loader(args.dataset, 'train', tokenizer, args.batch_size, shuffle=True)
    val_loader = get_data_loader(args.dataset, 'dev', tokenizer,32)
    logger.info("dataset {} loaded".format(args.dataset))
    logger.info("num_labels: {}".format(num_labels))
    model.train()
    model.to(device)
    pretrained_model = get_model(args)
    pretrained_model.to(device)
    if torch.cuda.device_count() > 0:
        pretrained_model = DataParallel(pretrained_model, device_ids=[int(i) for i in args.device.split(',')])
    pretrained_model.eval()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    logger.info('starting training')

    best_acc = 0
    step_counter = 0
    for epoch in range(args.epochs):
        model.train()
        pretrained_model.eval()
        logger.info("epoch {} start".format(epoch))
        start_time = datetime.now()
        step_counter = train_common_yake(model, optimizer, train_loader,
                                    epoch, log_steps=args.log_step, pre_model=pretrained_model,
                                    shift_reg=args.shift_reg,
                                    loss_type=args.loss_type, scl_reg=args.scl_reg,
                                    save_steps=args.save_steps, save_dir=output_dir, step_counter=step_counter)
        acc,f1,recall= test_acc(model, val_loader, args.eval_metric)
        logger.info("epoch {} validation {}: {:.4f} ".format(epoch, args.eval_metric, acc))
        logger.info("epoch {} validation_f1: {:.4f} ".format(epoch, f1))
        logger.info("epoch {} val_recall:{:.4f},val_human_recall:{:.4f},machine_recall:{:.4f} ".format(epoch,recall))

        if acc > best_acc:
            best_acc = acc
            logger.info("best validation {} improved to {:.4f}".format(
                args.eval_metric, best_acc))
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), '{}/{}'.format(output_dir, output_name))
            logger.info("model saved to {}/{}".format(output_dir, output_name))
        if args.save_every_epoch:
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), '{}/epoch{}_{}'.format(output_dir, epoch, output_name))
            logger.info("model saved to {}/epoch{}_{}".format(output_dir, epoch, output_name))
        end_time = datetime.now()
        logger.info('time for one epoch: {}'.format(end_time - start_time))
def eval(output_name='model.pt', data_name='test', name='train'):
    args = args_parser()
    # args setting
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
    return acc, f1

if __name__ == '__main__':
    args = args_parser()
    main()
    eval(output_name=args.output_name, data_name='test', name='train')



