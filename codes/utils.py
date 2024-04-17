import torch
import numpy as np
import os
import random
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

def test_verbalizer(pred):
  text_label = pred.lower()
    # try: 
    #     # text_label = pred.split('Relation : ')[1].strip()
    #     text_label = pred.split()[-1].strip()
    #     # text_label = re.findall("\nRelation: (.*)", pred)[-1]
    # except:
    #     text_label = pred    
  class_label = '0'
  if 'true' in text_label or 'positive' in text_label or 'yes' in text_label:
      class_label = '1'
  #print (f'PRED: {text_label} {class_label}')
  return text_label, class_label

def eval_verbalizer(pred):
  text_label = pred.lower()
  class_label = 0
  if 'true' in text_label or 'positive' in text_label or 'yes' in text_label:
      class_label = 1
  return text_label, class_label

def folder_check(mpath):
  if os.path.isdir(mpath): print (f'Checkpoint path: {mpath}')
  else: os.makedirs(mpath, exist_ok=True)

def print_args(params):
  logger.info(" **************** CONFIGURATION **************** ")
  for key, val in vars(params).items():
    key_str = "{}".format(key) + (" " * (30 - len(key)))
    logger.info("%s =   %s", key_str, val)

def count_parameters(model: torch.nn.Parameter):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = '{}'.format(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def set_deterministic():
    torch.cuda.empty_cache()
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def save_result(model, checkpoint_path, y_true, y_pred, f1_bi, f1_mi, pred_label):
    folder_check(checkpoint_path)
    with open(checkpoint_path+'/test_scores.txt', 'w') as out:
        out.write('f1_mi:'+str(f1_mi)+'\n')
        out.write('f1_bi:'+str(f1_bi)+'\n')
        out.write ('TRUE\tPRED\tORIG_PRED\n')
        for true, pred, text_pred in zip(y_true, y_pred, pred_label):
            out.write (str(true)+'\t'+str(pred)+'\t'+text_pred+'\n')
    model.save_pretrained(checkpoint_path)

def save_result_conv(model, checkpoint_path, y_true, y_pred, f1_bi):
    folder_check(checkpoint_path)
    with open(checkpoint_path+'/test_scores.txt', 'w') as out:
        out.write('f1_bi:'+str(f1_bi)+'\n')
        out.write ('TRUE\tPRED\n')
        for true, pred in zip(y_true, y_pred):
            out.write (str(true)+'\t'+str(pred)+'\n')
    model.save_pretrained(checkpoint_path)
