from transformers import AutoTokenizer, AutoModelForSequenceClassification, BloomTokenizerFast, BloomForSequenceClassification, BloomConfig
from peft import get_peft_model, PromptTuningConfig, PromptTuningInit, TaskType
import torch
import os, json, random, logging, argparse, math
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset
import evaluate
from transformers.optimization import Adafactor

from utils import *

set_deterministic()
device = "cuda"

def _get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=203)
  parser.add_argument("--input_path", type=str, default='')
  parser.add_argument("--checkpoint_path", type=str, default="checkpoints/")
  parser.add_argument("--model_name_or_config_path", type=str)
  parser.add_argument('--max_length', type=int, default=256)
  parser.add_argument('--is_full', type=int, default=0)
  parser.add_argument('--train_batch_size', type=int, default=8)
  parser.add_argument("--eval_batch_size", type=int, default=8)
  parser.add_argument("--epoch", type=int, default=10)
  parser.add_argument('--freeze_lm_emb', action='store_true', help='whether freezing lm parameters')
  parser.add_argument("--lr", type=float, default=3e-5)
  parser.add_argument("--warmup_proportion", type=float, default=0.06)
  parser.add_argument('--save_result', action='store_true')
  parser.add_argument('--multi_gpu', action='store_true')
#   parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

  return parser

def main():
    args = _get_parser().parse_args()
    set_seed(args.seed)
    if args.save_result: 
        folder_check(args.checkpoint_path)
        log_file = args.checkpoint_path+"/trainer.log"
    else: log_file = "trainer.log"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, 'w+'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    metric = evaluate.load("f1")
    train_config = vars(args)
    if args.save_result: 
        with open(os.path.join(args.checkpoint_path, f"train_config.json"), "w") as fp:
            json.dump(train_config, fp, indent=2, ensure_ascii=False)

    logging.info(f'\nPreprocess the input.. {args.input_path}')
    if args.is_full==1: 
        data_files = {"train": args.input_path+"/train_full.tsv", "test": args.input_path+"/test.tsv"}
        if 'ddi' in args.input_path:
            data_files = {"train": "/home/yuni/prompt/causal-prompt/dataset/original/ddi/train_full.tsv", "test": args.input_path+"/test.tsv"}
        logging.info(f'Using full data: True')
    else: data_files = {"train": args.input_path+"/train.tsv", "test": args.input_path+"/test.tsv"}
    dataset = load_dataset("csv", data_files=data_files, delimiter='\t')
    dataset = dataset.map( lambda x: {"labels": [int(label) for label in x["label"]]}, batched=True, num_proc=1)
    logging.info (f'{args.multi_gpu=} {args.save_result=} {args.lr=}')
    logging.info (dataset['test'][1])

    if any(k in args.model_name_or_config_path for k in ("gpt", "opt", "bloom")): padding_side = "left"
    else: padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_config_path, padding_side=padding_side)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_conventional(examples):
        # inps = [f"classify: {x}" for x in examples["sentence"]]
        # tokenized_input = tokenizer(inps, truncation=True, padding="max_length", max_length=args.max_length)
        tokenized_input = tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=args.max_length)
        return tokenized_input

    tokenized_datasets = dataset.map(
        tokenize_conventional,
        batched=True,
        # remove_columns=dataset["train"].column_names,
        remove_columns=["e1", "e2", "label"], #tsv
        load_from_cache_file=False,
        desc="Preprocess" )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]
    # logging.info (train_dataset[0])

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.eval_batch_size)

    if 'bloom' in args.model_name_or_config_path:
        model = BloomForSequenceClassification.from_pretrained(args.model_name_or_config_path, num_labels=2)
    else: #for gpt
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_config_path, return_dict=True)
        model.config.pad_token_id = tokenizer.pad_token_id
        # model.resize_token_embeddings(len(tokenizer))
    logging.info (f'(ORIG) trainable parameters: {count_parameters(model)}')

    logger.info(f'Freezing the LM: {args.freeze_lm_emb}')
    if args.freeze_lm_emb:
        for name, param in model.named_parameters():
            if 'gpt' in args.model_name_or_config_path:
                if 'transformer.wte' in name or 'transformer.wpe' in name : param.requires_grad = False
            elif 't5' in args.model_name_or_config_path:
                if 'classification_head' in name : param.requires_grad = True
                else: param.requires_grad = False
            elif 'bloom' in args.model_name_or_config_path:
                if 'transformer.word_embeddings' in name: param.requires_grad = False #for bloomz
            else:
                # param.requires_grad = False
                continue
            # if param.requires_grad: logging.info (f'{name} {param.requires_grad}')
            # else: logging.info (f'{name} False/no grad update')
    
    logging.info (f'(END) trainable parameters: {count_parameters(model)}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # if 't5' in args.model_name_or_config_path:
    #     optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps= args.warmup_proportion * (len(train_dataloader) * args.epoch),
        num_training_steps=(len(train_dataloader) * args.epoch),
    )

    model.to(device)
    best_f1 = 0.0

    for epoch in range(args.epoch):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'Training (ep={epoch})')):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        ep_pred, ep_ref = [], []
        for step, batch in enumerate(tqdm(eval_dataloader, desc=f'Testing (ep={epoch})')):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            # predicted_class_id = logits.argmax().item() #out: 1 or 0
            predictions = outputs.logits.argmax(dim=-1) #out tensor([0], device='cuda')
            predictions = predictions.detach().cpu().tolist()
            references = batch["labels"].detach().cpu().tolist()
            ep_pred.extend(predictions)
            ep_ref.extend(references)
            metric.add_batch(
                predictions=predictions,
                references=references, )

        eval_metric = metric.compute(pos_label=1, average="binary")
        # eval_metric = metric.compute(pos_label=1, average="micro")
        ep_f1 = eval_metric['f1']
        logging.info (f"epoch {epoch} {ep_f1}")
        if ep_f1 > best_f1:
            best_f1 = ep_f1
            logging.info (f'New test best f1: {ep_f1}')
            if args.save_result: save_result_conv(model, args.checkpoint_path, ep_ref, ep_pred, best_f1)

    logging.info (f'BEST TEST: {best_f1}\n')
    #logging.info (best_pred, best_ref)

if __name__ == '__main__':
  main()
