import numpy as np
import torch
from collections import defaultdict

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

def load_ner(path, tag2i, maxlen = 128):
    sents = []
    with open(path, encoding = "ISO-8859-1") as file:
        sent = []
        labels = []
        for line in file:
            line = line.split()
            if len(line) == 0:
                if len(sent) > 1 and len(sent) <= maxlen:
                    sents.append({'sent': sent, 'labels': labels})
                sent = []
                labels = []
            else:
                sent.append(line[0])
                labels.append(tag2i[line[-1]])
    return sents

def build_vocab(data):
    corpus = set()
    for tok in data:
        corpus.add(tok)
    return Vocab.from_corpus(corpus)

class Vocab: # 0 is reserved for padding
    def __init__(self, w2i):
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.items()}

    @classmethod
    def from_corpus(cls, corpus):
        w2i = {}
        for word in corpus:
            w2i.setdefault(word, len(w2i))
        return Vocab(w2i)

    def __len__(self):
        return len(self.w2i.keys())
    
    def insert(self, word):
        if word not in self.w2i:
            self.w2i[word] = len(self.w2i)
            self.i2w[len(self.i2w)] = word
            return True
        return False

def isint(string):
    try:
        int(string)
        return True
    except Exception:
        return False

def load_conllu(fname, maxlen = 128):
    # note: id starts from 1.
    columns = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel',
               'deps', 'misc']
    with open(fname) as file:
        a = []
        toks = []
        for line in file:
            if not line.strip():
                if len(toks) <= maxlen:
                    a.append(toks)
                else:
                    print("Removed length {}".format(len(toks)))
                toks = []
            elif not line.startswith('#'):
                tok = {}
                i = 0
                while '\t' in line:
                    entry, line = line[:line.index('\t')], line[line.index('\t')+1:]
                    tok[columns[i]] = entry
                    i += 1
                tok[columns[i]] = line[:line.index('\n')]
                if isint(tok['id']):
                    toks.append(tok)
                else:
                    pass # print("Skipping token {}".format(tok))
    return a

def sent_avgs(correct, pad_mask):
    # correct and padmask are same shape (num_sent x pad_len)
    # correct is 1 if correct, unknown for padding tokens
    # need to remove non-padding tokens, as well as cls and sep
    # we take the avg over each sentence, then sum the averages together
    # (at the end, we will divide by numsentences to get the macro-avg)
    pad_mask_2 = np.zeros(pad_mask.shape)
    pad_mask_2[:,:-1] = pad_mask[:,1:] # shift left by 1 to remove sep
    pad_mask_2[:,0] = 0 # remove cls
    total = 0
    for i in range(len(correct)):
        total += np.sum(correct[i] * pad_mask_2[i]) / np.sum(pad_mask_2[i])
    return total

def masked_loss(gold, pred, mask):
    # gold: num_sent x pad_len
    # pred: num_sent x pad_len x vocab
    # mask: num_sent x pad_len
    _, _, vocab_size = pred.shape
    loss = torch.nn.CrossEntropyLoss()
    #print("Pre-mask", gold.shape, pred.shape)
    gold = gold.masked_select(mask.byte())
    pred = pred.masked_select(mask.unsqueeze(-1).byte()).reshape(-1, vocab_size)
    #print("Post-mask", gold.shape, pred.shape)
    return loss(pred, gold)

def split_tag(chunk_tag):
    """
    split chunk tag into IOBES prefix and chunk_type
    e.g. 
    B-PER -> (B, PER)
    O -> (O, None)
    """
    if chunk_tag == 'O':
        return ('O', None)
    return chunk_tag.split('-', maxsplit=1)

def is_chunk_end(prev_tag, tag):
    """
    check if the previous chunk ended between the previous and current word
    e.g. 
    (B-PER, I-PER) -> False
    (B-LOC, O)  -> True

    Note: in case of contradicting tags, e.g. (B-PER, I-LOC)
    this is considered as (B-PER, B-LOC)
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 == 'O':
        return False
    if prefix2 == 'O':
        return prefix1 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']

def is_chunk_start(prev_tag, tag):
    """
    check if a new chunk started between the previous and current word
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 == 'O':
        return False
    if prefix1 == 'O':
        return prefix2 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']


def calc_metrics(tp, p, t, percent=True):
    """
    compute overall precision, recall and FB1 (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    precision = tp / p if p else 0
    recall = tp / t if t else 0
    fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * fb1
    else:
        return precision, recall, fb1


def count_chunks(true_seqs, pred_seqs):
    """
    true_seqs: a list of true tags
    pred_seqs: a list of predicted tags

    return: 
    correct_chunks: a dict (counter), 
                    key = chunk types, 
                    value = number of correctly identified chunks per type
    true_chunks:    a dict, number of true chunks per type
    pred_chunks:    a dict, number of identified chunks per type

    correct_counts, true_counts, pred_counts: similar to above, but for tags
    """
    correct_chunks = defaultdict(int)
    true_chunks = defaultdict(int)
    pred_chunks = defaultdict(int)

    correct_counts = defaultdict(int)
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    prev_true_tag, prev_pred_tag = 'O', 'O'
    correct_chunk = None

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):
        if true_tag == pred_tag:
            correct_counts[true_tag] += 1
        true_counts[true_tag] += 1
        pred_counts[pred_tag] += 1

        _, true_type = split_tag(true_tag)
        _, pred_type = split_tag(pred_tag)

        if correct_chunk is not None:
            true_end = is_chunk_end(prev_true_tag, true_tag)
            pred_end = is_chunk_end(prev_pred_tag, pred_tag)

            if pred_end and true_end:
                correct_chunks[correct_chunk] += 1
                correct_chunk = None
            elif pred_end != true_end or true_type != pred_type:
                correct_chunk = None

        true_start = is_chunk_start(prev_true_tag, true_tag)
        pred_start = is_chunk_start(prev_pred_tag, pred_tag)

        if true_start and pred_start and true_type == pred_type:
            correct_chunk = true_type
        if true_start:
            true_chunks[true_type] += 1
        if pred_start:
            pred_chunks[pred_type] += 1

        prev_true_tag, prev_pred_tag = true_tag, pred_tag
    if correct_chunk is not None:
        correct_chunks[correct_chunk] += 1

    return (correct_chunks, true_chunks, pred_chunks, 
        correct_counts, true_counts, pred_counts)

def get_result(correct_chunks, true_chunks, pred_chunks,
    correct_counts, true_counts, pred_counts, verbose=True):
    """
    if verbose, print overall performance, as well as preformance per chunk type;
    otherwise, simply return overall prec, rec, f1 scores
    """
    # sum counts
    sum_correct_chunks = sum(correct_chunks.values())
    sum_true_chunks = sum(true_chunks.values())
    sum_pred_chunks = sum(pred_chunks.values())

    sum_correct_counts = sum(correct_counts.values())
    sum_true_counts = sum(true_counts.values())

    nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != 'O')
    nonO_true_counts = sum(v for k, v in true_counts.items() if k != 'O')

    chunk_types = sorted(list(set(list(true_chunks) + list(pred_chunks))))

    # compute overall precision, recall and FB1 (default values are 0.0)
    prec, rec, f1 = calc_metrics(sum_correct_chunks, sum_pred_chunks, sum_true_chunks)
    res = (prec, rec, f1)
    if not verbose:
        return res

    # print overall performance, and performance per chunk type
    
    #print("processed %i tokens with %i phrases; " % (sum_true_counts, sum_true_chunks), end='')
    #print("found: %i phrases; correct: %i.\n" % (sum_pred_chunks, sum_correct_chunks), end='')
        
    #print("accuracy: %6.2f%%; (non-O)" % (100*nonO_correct_counts/nonO_true_counts))
    #print("accuracy: %6.2f%%; " % (100*sum_correct_counts/sum_true_counts), end='')
    #print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" % (prec, rec, f1))

    # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
    for t in chunk_types:
        prec, rec, f1 = calc_metrics(correct_chunks[t], pred_chunks[t], true_chunks[t])
        #print("%17s: " %t , end='')
        #print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
        #            (prec, rec, f1), end='')
        #print("  %d" % pred_chunks[t])

    return res
    # you can generate LaTeX output for tables like in
    # http://cnts.uia.ac.be/conll2003/ner/example.tex
    # but I'm not implementing this

def evaluate(true_seqs, pred_seqs, verbose=True):
    (correct_chunks, true_chunks, pred_chunks,
        correct_counts, true_counts, pred_counts) = count_chunks(true_seqs, pred_seqs)
    result = get_result(correct_chunks, true_chunks, pred_chunks,
        correct_counts, true_counts, pred_counts, verbose=verbose)
    return result
