import os, glob, json
import statistics
import argparse
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
import numpy

def get_args():
  options = argparse.ArgumentParser()
  options.add_argument('--folder', type=str, default='checkpoints/')
  options.add_argument('--cv', type=str, default='')
  options.add_argument('--tag', type=str, default='')
  options.add_argument('--filescore', type=str, default='test_scores.txt')
  options.add_argument('--idex', type=str, default='binary')
  return options.parse_args()

def mean_stdev(l):
  return {'mean':statistics.fmean(l), 'stdev': statistics.stdev(l)}

def calc(fnamelist, avg):
  result = {}
  result['p'] = []
  result['r'] = []
  result['f'] = []
  result['file'] = []
  result['len'] = []
  for filename in fnamelist:
    # print (filename)
    true, pred = [], []
    with open(filename, 'r') as fp:
      for i, line in enumerate(fp):
        if i > 2:
          xx = line.split('\t')
          true.append(float(xx[0]))
          pred.append(float(xx[1]))
    assert len(true) == len(pred)
    p,r,f,s = precision_recall_fscore_support(numpy.array(true), numpy.array(pred), average=avg)
    result['p'].append(p)
    result['r'].append(r)
    result['f'].append(f)
    result['file'].append(filename)
    result['len'].append(len(true))
  print (f"{result=}/n")
  return result

def get_fnamelist(folder, cv, idex, filescore):
  fnamelist = []
  for i in cv.split(','):
      fnamelist.append(folder+i+'/'+idex+'/test_scores.txt')
  # print(fnamelist)
  return fnamelist

def get_stats(p,r,f):
  ms_p = mean_stdev(p)
  ms_r = mean_stdev(r)
  ms_f = mean_stdev(f)
  print (f'{ms_p=} {ms_r=} {ms_f=}')

args = get_args()
fnamelist = get_fnamelist(args.folder, args.cv, args.idex, args.filescore)
result = calc(fnamelist, args.avg)
get_stats(result['p'], result['r'], result['f'])

# python get_score.py --folder k-shot/comagc/ --cv pathbn116-2,pathbn116-4,pathbn116-5,pathbn116-7,pathbn116-10 --idex bioroberta-pr6-pathbn1l3
