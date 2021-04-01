import sys
import checklist
from checklist.perturb import Perturb
import pandas as pd
import numpy as np
import spacy
import pattern
import collections


def add_negation(doc):
  """Adds negation to doc
  This is experimental, may or may not work. It also only works for specific parses.
  Parameters
  ----------
  doc : spacy.token.Doc
      input
  Returns
  -------
  string
      With negations added
  """
  for sentence in doc.sents:
    if len(sentence) < 3:
      continue
    root_id = [x.i for x in sentence if x.dep_ == 'ROOT'][0]
    root = doc[root_id]
    if '?' in sentence.text and sentence[0].text.lower() == 'how':
      continue
    if root.lemma_.lower() in ['thank', 'use']:
      continue
    if root.pos_ not in ['VERB', 'AUX']:
      continue
    neg = [True for x in sentence if x.dep_ == 'neg' and x.head.i == root_id]
    if neg:
      continue
    if root.lemma_ == 'be':
      if '?' in sentence.text:
        continue
      if root.text.lower() in ['is', 'was', 'were', 'am', 'are', '\'s', '\'re', '\'m']:
        return doc[:root_id + 1].text + ' not ' + doc[root_id + 1:].text
      else:
        return doc[:root_id].text + ' not ' + doc[root_id:].text
    else:
      aux = [x for x in sentence if x.dep_ in ['aux', 'auxpass'] and x.head.i == root_id]
      if aux:
        aux = aux[0]
        if aux.lemma_.lower() in ['can', 'do', 'could', 'would', 'will', 'have', 'should']:
          lemma = doc[aux.i].lemma_.lower()
          if lemma == 'will':
            fixed = 'won\'t'
          elif lemma == 'have' and doc[aux.i].text in ['\'ve', '\'d']:
            fixed = 'haven\'t' if doc[aux.i].text == '\'ve' else 'hadn\'t'
          elif lemma == 'would' and doc[aux.i].text in ['\'d']:
            fixed = 'wouldn\'t'
          else:
            fixed = doc[aux.i].text.rstrip('n') + 'n\'t' if lemma != 'will' else 'won\'t'
          fixed = ' %s ' % fixed
          return doc[:aux.i].text + fixed + doc[aux.i + 1:].text
        return doc[:root_id].text + ' not ' + doc[root_id:].text
      else:
        # TODO: does, do, etc. Remover return None de cima
        subj = [x for x in sentence if x.dep_ in ['csubj', 'nsubj']]
        p = pattern.en.tenses(root.text)
        tenses = collections.Counter([x[0] for x in pattern.en.tenses(root.text)]).most_common(1)
        tense = tenses[0][0] if len(tenses) else 'present'
        params = [tense, 3]
        if p:
          tmp = [x for x in p if x[0] == tense]
          if tmp:
            params = list(tmp[0])
          else:
            params = list(p[0])
        if root.tag_ not in ['VBG']:
          do = pattern.en.conjugate('do', *params) + 'n\'t'
          new_root = pattern.en.conjugate(root.text, tense='infinitive')
        else:
          do = 'not'
          new_root = root.text
        return '%s %s %s %s' % (doc[:root_id].text, do, new_root, doc[root_id + 1:].text)

# Check args validity
if len(sys.argv) == 3:
    input_address = sys.argv[1]
    output_address = sys.argv[2]
    print(3)
elif len(sys.argv) == 2:
    input_address = sys.argv[1]
    output_address = 'atis_intent_augment.csv'
    print(2)
else:
    print('invalid number of args')
    assert False

# read data from input argument
atis = pd.read_csv(input_address, names=['intent', 'sequence'])
seq = atis['sequence'].to_numpy()
intent = atis['intent'].to_numpy()

# load Spacy English corpus
nlp = spacy.load('en_core_web_sm')

# Create Augmented Data
docs = list(nlp.pipe(seq, n_process=1))
ret_negation = Perturb.perturb(docs, add_negation, keep_original=False)
negation = np.array(ret_negation.data).flatten().tolist()

# Write augmented data to output address
df = pd.DataFrame({'intent': intent, 'sequence': negation})
df.to_csv(output_address)
