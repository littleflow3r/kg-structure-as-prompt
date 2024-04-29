import os, argparse
import sys
import time
import jsonlines as jl
import pandas as pd
import numpy as np
import re
from utils import *

from openai import OpenAI
api_key = "" #put your API key here
client = OpenAI(api_key=api_key)

# Function to translate int into str relation
relation_int2str = lambda x: "non-causal" if x==0 else ("causal" if x==1 else x) 

def prepare_prompt(n, training_data): 
  # n_data = [[training_data.iloc[i]['sentence'], training_data.iloc[i]['relation'],training_data.iloc[i]['e1'],training_data.iloc[i]['e2']] for i in range(0, training_data.shape[0])]
  n_data = [[training_data.iloc[i]['sentence'], training_data.iloc[i]['relation'],training_data.iloc[i]['e1'],training_data.iloc[i]['e2']] for i in np.random.randint(0, training_data.shape[0], size=n)]

  training_contexts = ""
  training_answers = ""
  for i, data in enumerate(n_data):
    sentence, relation, e1, e2 = data[0], data[1], data[2], data[3]
    training_contexts += f"Context #{i+1}: [{sentence}]\n"
    training_answers += f"Result #{i+1}: ['e1': '{e1}', 'relation': '{relation_int2str(relation)}', 'e2': '{e2}]'\n"
  return f"{training_contexts}\n{training_answers}\n"

def extract_relation(text: str):
   if "'relation': 'non-causal'" in text: return 0
   elif "'relation': 'causal'": return 1
   else: return "NA"

def send_prompt (text, model):
  # response = openai.Completion.create(
  response = client.completions.create(
    model=model,
    prompt=text,
    temperature=0.7,
    # max_tokens=400, #m7
    max_tokens=100, #m=2
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  return response.choices[0].text.lstrip()

def _get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_path", type=str, default='')
  parser.add_argument("--checkpoint_path", type=str, default="checkpoints/")
  parser.add_argument("--model", type=str)
  parser.add_argument('--k', type=str, default="false,true")
  parser.add_argument('--n_samples',  type=int, default=32)
  parser.add_argument('--m_samples', type=int, default=10)
  return parser

def main():
  args = _get_parser().parse_args()
  f_i = args.input_path
  f_o = args.checkpoint_path
  i_cv = args.k
  n_samples = args.n_samples
  m_samples = args.m_samples
  print (f'{f_i=} {f_o=} {i_cv=} {n_samples=} {m_samples=} {args.model=}')
  training_data = pd.read_json(f'{f_i}{i_cv}/train.jsonl', lines=True)

  try:  # create directory
    os.makedirs(f_o)
  except FileExistsError:
    pass

  n_rows = None
  with jl.open(f'{f_i}{i_cv}/test.jsonl', 'r') as f_test:
    n_rows = sum([1 for line in f_test.iter()])

  with jl.open(f'{f_i}{i_cv}/test.jsonl', 'r') as f_test, open(f'{f_o}result_{i_cv}.tsv', 'w') as f_out:
    f_out.write('Gold\t0/1_Response\tOriginal_Response\tContext\n')
    
    test_data = ""
    test_relations = []
    # test_ids = []
    instruction = "Given the context sentence, classify the relationship between the entities marked with e1 and e2 as 'causal' or 'non-causal' relation.\n\n"

    for i, row in enumerate(f_test):
      if (i+1) % m_samples == 0 or (i+1) == n_rows:
          test_data += f"Context #{(i%m_samples)+1}: [{row['sentence']}]\n"
          test_relations.append(row['relation'])
          # test_ids.append(row['id'])
          prompt = instruction + prepare_prompt(int(n_samples), training_data) + test_data #len= 2100
          print(f"PROMPT:\n\n{prompt}")
          
          # buffer = input("REPLY (press any key): ")
        #  print("REPLY:")
          reply = send_prompt(prompt, args.model)
          
        #  [print(l, relation_int2str(r)) for l, r in zip(str(reply).split("\n"), test_relations)]
          result = lambda p,c: f"\033[92m{p}\033[00m" if str(p).split("'relation': ")[1].split(", ")[0].strip("'") == relation_int2str(c) else f"\033[91m{p}\033[00m"
          # [print(result(p, c)) for p, c in zip(str(reply).split("\n"), test_relations)]
          for p, c in zip(str(reply).split("\n"), test_relations):
            try:
              print(result(p, c))
            except:
              pass
          # for t_id, r, c in zip(test_ids, str(reply).split("\n"), test_data.split("\n")):
          for r, c in zip(str(reply).split("\n"), test_data.split("\n")):
            try: 
              # f_out.write(f"{t_id}\t{row['relation']}\t{extract_relation(r)}\t{r}\t{str(c).split('[')[1].split(']')[0]}\n")
              f_out.write(f"{row['relation']}\t{extract_relation(r)}\t{r}\t{str(c).split('[')[1].split(']')[0]}\n")
            except:
              print(f"Couldn't write to file:\n")
          

          print("#"*100)
          # sys.exit()
          
          del(prompt)
          test_data = ""
          test_relations = []
          # test_ids = []
      else:
        test_data += f"Context #{(i%m_samples)+1}: [{row['sentence']}]\n"
        test_relations.append(row['relation'])
        # test_ids.append(row['id'])

  print()
  del(training_data)

if __name__ == '__main__':
  main()

