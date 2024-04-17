from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import get_peft_model, PromptTuningConfig, PromptTuningInit, TaskType
import torch
import os, json, random, logging, argparse, math
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset
import evaluate
from sklearn.metrics import f1_score, precision_recall_fscore_support

from utils import *

import warnings
warnings.filterwarnings("ignore")

set_deterministic()
device = "cuda"

def _get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=203)
  parser.add_argument("--input_path", type=str, default='')
  parser.add_argument("--checkpoint_path", type=str, default="checkpoints/")
  parser.add_argument("--model_name_or_config_path", type=str)
  parser.add_argument('--max_length', type=int, default=256)
  parser.add_argument('--with_evs', type=int, default=0)
  parser.add_argument('--text_label', type=str, default="false,true")
  parser.add_argument('--prompt_text', type=str, default="")
  parser.add_argument('--prompt_tuning_init', type=str, default="random")
  parser.add_argument('--prompt_tuning_init_text', type=str)
  parser.add_argument('--num_virtual_tokens', type=int, default=20)
  parser.add_argument('--train_batch_size', type=int, default=8)
  parser.add_argument("--eval_batch_size", type=int, default=8)
  parser.add_argument("--epoch", type=int, default=10)
#   parser.add_argument("--peft_on", action='store_true',)
#   parser.add_argument("--early_stopping", type=int, default=10)
  parser.add_argument('--freeze_lm_emb', action='store_false', help='whether freezing lm parameters')
  parser.add_argument("--lr", type=float, default=3e-5)
  parser.add_argument("--warmup_proportion", type=float, default=0.06)
  parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
  parser.add_argument("--max_grad_norm", type=float, default=1.0)
  parser.add_argument('--multi_gpu', action='store_true')
  parser.add_argument('--prompt_front', action='store_true')
  parser.add_argument('--save_result', action='store_true')
  return parser

def testing(model, tokenizer, test_dataset, target_max_length, max_length, prompt_text, with_evs):
    model.eval()
    y_true, y_pred = [], []
    pred_label = []
    for i in tqdm(range(0, len(test_dataset)), desc='Testing'):
        sentence = test_dataset[i]['sentence']
        evs = test_dataset[i]['evs']
        if evs is None: evs=' '
        if with_evs == 1:
            inps = evs+' '+sentence #3
        elif with_evs == 2:
            evs = tokenizer(evs)["input_ids"][:max_length-len(sentence)-1] #1
            evs = ' '.join(tokenizer.batch_decode(evs))
            inps = evs+' '+sentence #3
            # inps = ' '.join(evs.split())+' '+sentence
            # inps = ' '.join(evs.split()[:round(max_length/2)])+' '+' '.join(sentence.split()[:round(max_length/2)]) #2            
        else: 
            inps = sentence
            # inps = ' '.join(sentence.split()[:round(max_length/2)])
        inputs = tokenizer(f'{inps} {prompt_text}', return_tensors="pt")
        truth_label = test_dataset[i]['label']
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # outputs = model.module.generate( #multi_gpu
            outputs = model.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=target_max_length, eos_token_id=tokenizer.eos_token_id
            )
            pred = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
            ptext_label, pclass_label = test_verbalizer(pred)
            # print (f'PRED: {ptext_label} {pclass_label}')
            y_true.append(int(truth_label))
            y_pred.append(int(pclass_label))
            pred_label.append(ptext_label)
    f1_mi = f1_score(y_true, y_pred, average='micro')
    f1_bi = f1_score(y_true, y_pred, average='binary')
    alls = precision_recall_fscore_support(y_true, y_pred, average='binary')
    # f1_score(y_true, y_pred, average='binary')
    logging.info (f'F1 binary/micro: {f1_bi} / {f1_mi} \n')
    return y_true, y_pred, f1_bi, f1_mi, pred_label, alls

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

    logging.info (f'Input: {args.input_path}')
    data_files = {"train": args.input_path+"/train.tsv", "test": args.input_path+"/test.tsv"}
    dataset = load_dataset("csv", data_files=data_files, delimiter='\t')
    classes = args.text_label.split(',')
    logging.info (f'text label: {classes}')
    dataset = dataset.map( lambda x: {"text_label": [classes[int(label)] for label in x["label"]]}, batched=True, num_proc=1)
    logging.info (dataset['test'][1])
    logging.info (f'{args.with_evs=} {args.prompt_front=} {args.multi_gpu=} {args.save_result=} {args.lr=}')
    # if any(k in args.model_name_or_config_path for k in ("gpt", "opt", "bloom")): padding_side = "left"
    # else: padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_config_path, padding_side='left')
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])

    if args.prompt_tuning_init == 'text':
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            # num_virtual_tokens=len(tokenizer(args.prompt_tuning_init_text)["input_ids"]),
            num_virtual_tokens=args.num_virtual_tokens,
            prompt_tuning_init_text=args.prompt_tuning_init_text,
            tokenizer_name_or_path=args.model_name_or_config_path,
            num_layers=12,
            token_dim=768,
            # num_transformer_submodules=1,
            # num_attention_heads=12, 
            )
    else:
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=args.num_virtual_tokens,
            # num_layers=12, 
            )
    
    def preprocess_function(examples):
        batch_size = len(examples['sentence'])
        if args.with_evs == 1:
            evs = [' ' if v is None else v for v in examples['evs']]
            inps = [c+' '+a for c, a in zip(evs, examples['sentence'])]
        elif args.with_evs == 2:
            evs = [' ' if v is None else v for v in examples['evs']]
            evs = [tokenizer(e)["input_ids"][:args.max_length-len(a)-1] for e, a in zip(evs, examples['sentence'])]
            evs = [' '.join(tokenizer.batch_decode(e)) for e in evs]
            inps = [c+' '+a for c, a in zip(evs, examples['sentence'])]
            # inps = [' '.join(c.split()[:round(args.max_length/2)])+' '+' '.join(a.split()[:round(args.max_length/2)]) for c, a in zip(examples['evs'], examples['sentence'])]
        else: 
            inps = examples['sentence']
        # inputs = [f"Sentence : {x} Relation : " for x in examples['sentence']]
        inputs = [f"{x} {args.prompt_text}" for x in inps]
        if args.prompt_front:
            inputs = [f"{args.prompt_text} {x} " for x in inps]
        # targets = [str(x) for x in examples['text_label']]
        targets = examples['text_label']
        model_inputs = tokenizer(inputs)
        labels = tokenizer(targets, add_special_tokens=False)  # don't add bos token because we concatenate with inputs
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                args.max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (args.max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (args.max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:args.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:args.max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:args.max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        # remove_columns=dataset["train"].column_names,
        remove_columns=["e1", "e2"], #tsv
        load_from_cache_file=False,
        desc="Preprocess" )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]
    # logging.info (f"Sample input_ids/labels: {train_dataset[0]['input_ids']} {train_dataset[0]['labels']}")

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'label'], device=device)
    eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'label'], device=device)
    # print (train_dataset[0]['label'])

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.eval_batch_size)

    config = AutoConfig.from_pretrained(args.model_name_or_config_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_config_path, config=config)
    if 'gpt' in args.model_name_or_config_path:
        model.config.pad_token_id = tokenizer.pad_token_id 
    model = get_peft_model(model, peft_config)
    logging.info (f'(ORIG) trainable parameters: {count_parameters(model)}')

    logger.info(f'Freezing the LM: {args.freeze_lm_emb}')
    if args.freeze_lm_emb:
        for name, param in model.named_parameters():
            if 'transformer.word_embeddings' in name: param.requires_grad = False #for bloomz
            elif 'transformer.wte' in name or 'transformer.wpe' in name : param.requires_grad = False #for gpt
            else: param.requires_grad = True
            # if param.requires_grad: print (f'{name} {param.requires_grad}')
            # else: print (f'{name} False/no grad')

    logging.info (f'(END) trainable parameters: {count_parameters(model)}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps= args.warmup_proportion * (len(train_dataloader) * 20),
        num_training_steps=(len(train_dataloader) * 20),
    )

    if args.multi_gpu:
        model=torch.nn.DataParallel(model)

    model.to(device)
    test_best = 0.

    for epoch in range(args.epoch):
        model.train()
        total_loss = 0.
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'Training (ep={epoch})')):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.multi_gpu:
                loss.sum().backward()
            else:loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
            total_loss += loss.detach().float()

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        logging.info (f"{epoch=}: {train_ppl=} {train_epoch_loss=}")
        y_true, y_pred, f1_bi, f1_mi, pred_label, alls = testing(model, tokenizer, dataset["test"], target_max_length, args.max_length, args.prompt_text, args.with_evs)
        if f1_bi > test_best:
            test_best = f1_bi
            logging.info (f'Congratulations! New Test Best Accuracy Score: {test_best}, {alls}')
            if args.save_result: save_result(model, args.checkpoint_path, y_true, y_pred, f1_bi, f1_mi, pred_label)

        # model.eval()
        # eval_loss = 0
        # eval_preds, eval_preds_textlabel = [], []
        # for step, batch in enumerate(tqdm(eval_dataloader, desc=f'Evaluate (ep={epoch})')):
        #     batch = {k: v.to(device) for k, v in batch.items()}
        #     with torch.no_grad():
        #         outputs = model(**batch)
        #     loss = outputs.loss
        #     eval_loss += loss.detach().float()
        #     pred = tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)[0]
        #     ptext_label, pclass_label = eval_verbalizer_bloom(pred)
        #     eval_preds.extend([pclass_label])
        #     eval_preds_textlabel.extend([ptext_label])

        # eval_epoch_loss = eval_loss / len(eval_dataloader)
        # eval_ppl = torch.exp(eval_epoch_loss)
        # train_epoch_loss = total_loss / len(train_dataloader)
        # train_ppl = torch.exp(train_epoch_loss)
        # # logging.info (f"{epoch=}: {train_epoch_loss=} {eval_epoch_loss=}\n")
        
        # logging.info (f"{eval_preds_textlabel[:5]=}")
        # # logging.info (f"{eval_preds[:5]=}")
        # logging.info (f"{dataset['test']['text_label'][:5]=}")
        # # logging.info (f"{dataset['test']['label'][:5]=}")
        # f1_mi = f1_score(dataset["test"]["label"], eval_preds, average='micro')
        # f1_bi = f1_score(dataset["test"]["label"], eval_preds, average='binary')
        # logging.info (f'F1 binary/micro: {f1_bi} / {f1_mi} \n')

        # if f1_bi > test_best:
        #     test_best = f1_bi
        #     logging.info (f'Congratulations! New Test Best Accuracy Score: {test_best}')
        #     if args.save_result: save_result(model, args.checkpoint_path, dataset['test']['label'], eval_preds, f1_bi, f1_mi, eval_preds_textlabel)

    logging.info (f'BEST TEST: {test_best} Checkpoint: {args.checkpoint_path}\n\n')

if __name__ == '__main__':
  main()
