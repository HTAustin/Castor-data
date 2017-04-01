import re
import os
import numpy as np
# import cPickle
import subprocess
from collections import defaultdict
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import *

UNKNOWN_WORD_IDX = 0


def load_data(fname):
  lines = open(fname).readlines()
  qids, questions, answers, labels = [], [], [], []
  num_skipped = 0
  prev = ''
  qid2num_answers = {}
  for i, line in enumerate(lines):
    line = line.strip()

    qid_match = re.match('<QApairs id=\'(.*)\'>', line)

    if qid_match:
      qid = qid_match.group(1)
      qid2num_answers[qid] = 0

    if prev and prev.startswith('<question>'):
      question = line.lower().split('\t')

    label = re.match('^<(positive|negative)>', prev)
    if label:
      label = label.group(1)
      label = 1 if label == 'positive' else 0
      answer = line.lower().split('\t')
      if len(answer) > 60:
        # Truncate at max 60 tokens
        answer =  answer[0:60]
        # num_skipped += 1
        # continue
      labels.append(label)
      answers.append(answer)
      questions.append(question)
      qids.append(qid)
      qid2num_answers[qid] += 1
    prev = line

  return qids, questions, answers, labels




def add_to_vocab(data, alphabet):
  for sentence in data:
    for token in sentence:
      alphabet.add(token)


def convert2indices(data, alphabet, dummy_word_idx, max_sent_length=40):
  data_idx = []
  for sentence in data:
    ex = np.ones(max_sent_length) * dummy_word_idx
    for i, token in enumerate(sentence):
      idx = alphabet.get(token, UNKNOWN_WORD_IDX)
      ex[i] = idx
    data_idx.append(ex)
  data_idx = np.array(data_idx).astype('int32')
  return data_idx

def write_to_file(xmlF,outdir):
  
  if not os.path.exists(outdir):
    os.makedirs(outdir)

  
  qids, questions, answers, labels = load_data(xmlF)
  
  qid_file = open(outdir+'/id.txt','w')
  for qid in qids:
    qid_file.write(qid+'\n')
  qid_file.close()


  questions_file = open(outdir+'/a.toks','w')
  for q in questions:
    q_toks = TreebankWordTokenizer().tokenize(' '.join(q))
    q_str = ' '.join(q_toks).lower()
    questions_file.write(q_str+'\n')
  questions_file.close()


  answers_file = open(outdir+'/b.toks','w')
  for a in answers:
    a_toks = TreebankWordTokenizer().tokenize(' '.join(a))
    a_str = ' '.join(a_toks).lower()
    answers_file.write(a_str+'\n')
  answers_file.close()


  sim_file = open(outdir+'/sim.txt','w')
  for label in labels:
    sim_file.write(str(label)+'\n')
  sim_file.close()

  print(outdir+' dataset done!')



if __name__ == '__main__':

  stoplist = None


  train = 'jacana-qa-naacl2013-data-results/train.xml'
  write_to_file(train,'TrecQA/train')
  train_all = 'jacana-qa-naacl2013-data-results/train-all.xml'
  write_to_file(train_all,'TrecQA/train-all')  

  dev = 'jacana-qa-naacl2013-data-results/dev.xml'
  write_to_file(dev,"TrecQA/raw-dev")
  test = 'jacana-qa-naacl2013-data-results/test.xml'
  write_to_file(test,"TrecQA/raw-test")

    
