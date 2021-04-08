import torch
import torch.nn.functional as F
import numpy as np

from util import from_numpy, partial_state_dict
from classifiers import POSModel, NERModel, UDModel
from datasets import load_conllu, build_vocab, sent_avgs, masked_loss, evaluate, load_ner

def train_pos(bert_encoder, train_path, dev_path, lambda_init = 1000, lambda_final = 10000,
              lr_base = 3e-5, mask_lr_base = 0.1, lr_warmup_frac = 0.1,
              epochs = 3, batch_size = 32, verbose = True,
              lambda_startup_frac = 0.25, lambda_warmup_frac = 0.5, subbatch_size = 8,
              masked = True):
    def calc_dev(): 
        model.eval()
        language_acc = 0
        for j in range(0, len(dev_data), batch_size):
            data = dev_data[j:j+batch_size]
            sentences, pos_tags, pad_mask = data_to_padded(data, pos_vocab)
            with torch.no_grad():
                pos_pred = model.predict_batch(sentences)
                pos_pred = pos_pred.cpu().numpy()
            pos_correct = np.argmax(pos_pred, axis = 2) == pos_tags
            language_acc += sent_avgs(pos_correct, pad_mask)
        return language_acc / len(dev_data)

    def data_to_padded(data, pos_vocab):
        sentences = [[tok['form'] for tok in exmp] for exmp in data]
        pos_tags = [[pos_vocab.w2i[tok['upos']] for tok in exmp] for exmp in data]
        # need to convert to padded
        lengths = [len(s) + 2 for s in sentences]
        pos_tags_padded = np.zeros((len(sentences), max(lengths)), dtype = int)
        pad_mask = np.zeros((len(sentences), max(lengths)), dtype = int)
        for i, length in enumerate(lengths):
            pos_tags_padded[i, 1:length-1] = pos_tags[i]
            pad_mask[i, :length] = 1
        return sentences, pos_tags_padded, pad_mask

    train_data = load_conllu(train_path)
    dev_data = load_conllu(dev_path)
    pos_vocab = build_vocab(sum([[tok['upos'] for tok in exmp] for exmp in train_data], []))
    model = POSModel(bert_encoder, len(pos_vocab))

    print("lambda_init: {}, lambda_final: {}, lambda_startup_frac: {}, lambda_warmup_frac: {}".format(
        lambda_init, lambda_final, lambda_startup_frac, lambda_warmup_frac))
    print("lr_base: {}, mask_lr_base: {}, lr_warmup_frac: {}, epochs: {}, batch_size: {}".format(
        lr_base, mask_lr_base, lr_warmup_frac, epochs, batch_size))

    # group parameters for different learning rates
    mask_params = [p for n, p in model.named_parameters() if 'mask_score' in n and p.requires_grad]
    bert_params = [p for n, p in model.named_parameters() if 'mask_score' not in n and p.requires_grad]
    trainer = torch.optim.Adam([
        {'params': mask_params, 'lr': 0., 'lr_base': mask_lr_base, 'name': 'mask'},
        {'params': bert_params, 'lr': 0., 'lr_base': lr_base, 'name': 'bert'},], lr = 0.)
    
    def set_lr(lr_ratio):
        for param_group in trainer.param_groups:
            param_group['lr'] = param_group['lr_base'] * lr_ratio

    def converged(log, k = 10, thresh = 0.01, min_epochs = 25):
        if len(log) < min_epochs:
            return False
        accs = [l['dev_acc'] for l in log[-k:]]
        accs_converged = (np.max(accs) - np.min(accs)) < thresh
        if masked:
            sparsity = [l['reg_val'] for l in log[-k:]]
            sparsity_converged = (np.max(sparsity) - np.min(sparsity)) < thresh
            return accs_converged and sparsity_converged
        return accs_converged

    log = []
    processed = 0
    check_processed = 0
    check_every = 2048
    lambda_reg = lambda_init
    for epoch in range(epochs):#tqdm.notebook.tqdm(range(epochs)):
        np.random.shuffle(train_data)
        for i in range(0, len(train_data), batch_size):#tqdm.notebook.tqdm(range(0, len(train_data), batch_size)):
            examples = train_data[i:i+batch_size]
            model.train()
            trainer.zero_grad()
            if len(examples) == batch_size:
                # compute loss, also log other metrics
                for i in range(0, len(examples), subbatch_size):
                    examples_subbatch = examples[i:i+subbatch_size]
                    sentences, pos_tags, pad_mask = data_to_padded(examples_subbatch, pos_vocab)
                    pos_tags = from_numpy(pos_tags)
                    pad_mask = from_numpy(pad_mask)
                    pos_pred = model.predict_batch(sentences)
                    
                    loss = masked_loss(pos_tags, pos_pred, pad_mask)
                    (loss * len(examples_subbatch) / len(examples)).backward()
                if masked:
                    reg = model.bert.compute_total_regularizer()
                    (lambda_reg * reg).backward()
                #mask_grad_norm = torch.nn.utils.clip_grad_norm_(mask_params, np.inf)
                #bert_grad_norm = torch.nn.utils.clip_grad_norm_(bert_params, np.inf)
                trainer.step()

                processed += len(examples)
                check_processed += len(examples)

                # warmup from 0 to lr_base for lr_warmup_frac
                lr_ratio = min(1, processed / (lr_warmup_frac * epochs * len(train_data)))
                set_lr(lr_ratio)
                
                # schedule lambda_reg - constant, then linear, then constant
                lambda_ratio = max(0, min(1, (processed - lambda_startup_frac * epochs * len(train_data)) / (lambda_warmup_frac * epochs * len(train_data))))
                lambda_reg = lambda_init + (lambda_final - lambda_init) * lambda_ratio

                if check_processed >= check_every:
                    if masked:
                        log.append({'dev_acc': calc_dev(),
                                    'loss_val': loss.item(), 
                                    'reg_val': reg.item(),
                                    #'mask_grad_norm': mask_grad_norm.item(), 
                                    #'bert_grad_norm': bert_grad_norm.item(), 
                                    'pct_binary': model.bert.compute_binary_pct()})
                    else:
                        log.append({'dev_acc': calc_dev(),
                                    'loss_val': loss.item()})
                    check_processed -= check_every
                    if converged(log):
                        break
                    if verbose:
                        print("Log: {}".format(log[-1]))
        if converged(log):
            break
    return log, model

def train_ud(bert_encoder, train_path, dev_path, lambda_init = 1000, lambda_final = 10000,
             lr_base = 3e-5, mask_lr_base = 0.1, lr_warmup_frac = 0.1,
             epochs = 3, batch_size = 32, verbose = True,
             lambda_startup_frac = 0.25, lambda_warmup_frac = 0.5, subbatch_size = 8,
             masked = True):
    def calc_dev(): 
        model.eval()
        language_acc = 0
        for j in range(0, len(dev_data), batch_size):
            data = dev_data[j:j+batch_size]
            sentences, pos_tags, dep_idx, dep_rels, pad_mask = data_to_padded(data, pos_vocab, deprel_vocab)
            with torch.no_grad():
                pos_pred, idx_pred, deprel_pred = model.predict_batch(sentences, from_numpy(dep_idx))
                pos_pred = pos_pred.cpu().numpy()
                idx_pred = idx_pred.cpu().numpy()
                deprel_pred = deprel_pred.cpu().numpy()
            # pos_correct = np.argmax(pos_pred, axis = 2) == pos_tags
            idx_correct = np.argmax(idx_pred, axis = 2) == dep_idx
            deprel_correct = np.argmax(deprel_pred, axis = 2) == dep_rels
            all_correct = idx_correct * deprel_correct
            language_acc += sent_avgs(all_correct, pad_mask)
        return language_acc / len(dev_data)

    def data_to_padded(data, pos_vocab, deprel_vocab):
        sentences = [[tok['form'] for tok in exmp] for exmp in data]
        pos_tags = [[pos_vocab.w2i[tok['upos']] for tok in exmp] for exmp in data]
        dep_idx = [[int(tok['head']) for tok in exmp] for exmp in data]
        dep_rels = [[deprel_vocab.w2i[tok['deprel']] for tok in exmp] for exmp in data]
        
        # need to convert to padded
        lengths = [len(s) + 2 for s in sentences]
        pos_tags_padded = np.zeros((len(sentences), max(lengths)), dtype = int)
        dep_idx_padded = np.zeros((len(sentences), max(lengths)), dtype = int)
        dep_rels_padded = np.zeros((len(sentences), max(lengths)), dtype = int)
        pad_mask = np.zeros((len(sentences), max(lengths)), dtype = int)
        for i, length in enumerate(lengths):
            pos_tags_padded[i, 1:length-1] = pos_tags[i]
            dep_idx_padded[i, 1:length-1] = dep_idx[i]
            dep_rels_padded[i, 1:length-1] = dep_rels[i]
            pad_mask[i, :length] = 1
        return sentences, pos_tags_padded, dep_idx_padded, dep_rels_padded, pad_mask

    def converged(log, k = 10, thresh = 0.01, min_epochs = 25):
        if len(log) < min_epochs:
            return False
        accs = [l['dev_acc'] for l in log[-k:]]
        accs_converged = (np.max(accs) - np.min(accs)) < thresh
        if masked:
            sparsity = [l['reg_val'] for l in log[-k:]]
            sparsity_converged = (np.max(sparsity) - np.min(sparsity)) < thresh
            return accs_converged and sparsity_converged
        return accs_converged

    train_data = load_conllu(train_path)
    dev_data = load_conllu(dev_path)
    pos_vocab = build_vocab(sum([[tok['upos'] for tok in exmp] for exmp in train_data], []))
    deprel_vocab = build_vocab(sum([[tok['deprel'] for tok in exmp] for exmp in train_data], []))
    model = UDModel(bert_encoder, len(pos_vocab), len(deprel_vocab))

    print("lambda_init: {}, lambda_final: {}, lambda_startup_frac: {}, lambda_warmup_frac: {}".format(
        lambda_init, lambda_final, lambda_startup_frac, lambda_warmup_frac))
    print("lr_base: {}, mask_lr_base: {}, lr_warmup_frac: {}, epochs: {}, batch_size: {}".format(
        lr_base, mask_lr_base, lr_warmup_frac, epochs, batch_size))

    # group parameters for different learning rates
    mask_params = [p for n, p in model.named_parameters() if 'mask_score' in n and p.requires_grad]
    bert_params = [p for n, p in model.named_parameters() if 'mask_score' not in n and p.requires_grad]
    trainer = torch.optim.Adam([
        {'params': mask_params, 'lr': 0., 'lr_base': mask_lr_base, 'name': 'mask'},
        {'params': bert_params, 'lr': 0., 'lr_base': lr_base, 'name': 'bert'},], lr = 0.)
    
    def set_lr(lr_ratio):
        for param_group in trainer.param_groups:
            param_group['lr'] = param_group['lr_base'] * lr_ratio

    log = []
    processed = 0
    check_processed = 0
    check_every = 2048
    lambda_reg = lambda_init
    for epoch in range(epochs):#tqdm.notebook.tqdm(range(epochs)):
        np.random.shuffle(train_data)
        for i in range(0, len(train_data), batch_size):#tqdm.notebook.tqdm(range(0, len(train_data), batch_size)):
            examples = train_data[i:i+batch_size]
            model.train()
            trainer.zero_grad()
            if len(examples) == batch_size:
                # compute loss, also log other metrics
                for i in range(0, len(examples), subbatch_size):
                    examples_subbatch = examples[i:i+subbatch_size]
                    sentences, pos_tags, dep_idx, dep_rels, pad_mask = data_to_padded(examples_subbatch, pos_vocab, deprel_vocab)
                    # pos_tags = from_numpy(pos_tags)
                    dep_idx = from_numpy(dep_idx)
                    dep_rels = from_numpy(dep_rels)
                    pad_mask = from_numpy(pad_mask)
                    pos_pred, idx_pred, deprel_pred = model.predict_batch(sentences, dep_idx)
                    
                    loss = masked_loss(dep_idx, idx_pred, pad_mask) + masked_loss(dep_rels, deprel_pred, pad_mask)
                    (loss * len(examples_subbatch) / len(examples)).backward()
                if masked:
                    reg = model.bert.compute_total_regularizer()
                    (lambda_reg * reg).backward()
                #mask_grad_norm = torch.nn.utils.clip_grad_norm_(mask_params, np.inf)
                #bert_grad_norm = torch.nn.utils.clip_grad_norm_(bert_params, np.inf)
                trainer.step()

                processed += len(examples)
                check_processed += len(examples)

                # warmup from 0 to lr_base for lr_warmup_frac
                lr_ratio = min(1, processed / (lr_warmup_frac * epochs * len(train_data)))
                set_lr(lr_ratio)
                
                # schedule lambda_reg - constant, then linear, then constant
                lambda_ratio = max(0, min(1, (processed - lambda_startup_frac * epochs * len(train_data)) / (lambda_warmup_frac * epochs * len(train_data))))
                lambda_reg = lambda_init + (lambda_final - lambda_init) * lambda_ratio

                if check_processed >= check_every:
                    if masked:
                        log.append({'dev_acc': calc_dev(),
                                    'loss_val': loss.item(), 
                                    'reg_val': reg.item(),
                                    #'mask_grad_norm': mask_grad_norm.item(), 
                                    #'bert_grad_norm': bert_grad_norm.item(), 
                                    'pct_binary': model.bert.compute_binary_pct()})
                    else:
                        log.append({'dev_acc': calc_dev(),
                                    'loss_val': loss.item()})
                    check_processed -= check_every
                    if converged(log):
                        break
                    if verbose:
                        print("Log: {}".format(log[-1]))
        if converged(log):
            break
    return log, model

def train_ner(bert_encoder, train_path, dev_path, lambda_init = 1000, lambda_final = 10000,
              lr_base = 3e-5, mask_lr_base = 0.1, lr_warmup_frac = 0.1,
              epochs = 3, batch_size = 32, verbose = True,
              lambda_startup_frac = 0.25, lambda_warmup_frac = 0.5, subbatch_size = 8,
              masked = False):
    def calc_dev(): 
        model.eval()
        all_preds = np.array([])
        all_labels = pack_labels([exmp['labels'] for exmp in dev_data])
        for j in range(0, len(dev_data), batch_size):
            exmps = dev_data[j:j+batch_size]
            sents = [exmp['sent'] for exmp in exmps]
            with torch.no_grad():
                pred = model.predict_batch(sents).cpu().numpy() # numsent x 3
            pred = np.argmax(pred, axis = 1)
            all_preds = np.concatenate((all_preds, pred))
        return calc_f1(all_preds, all_labels)

    def pack_labels(labels):
        return np.array(sum(labels, []))

    def calc_f1(preds, labels):
        assert len(preds) == len(labels)
        preds = [model.i2tag[pred] for pred in preds]
        labels = [model.i2tag[label] for label in labels]
        result = evaluate(labels, preds)
        return result[2] / 100

    def converged(log, k = 10, thresh = 0.01, min_epochs = 25):
        if len(log) < min_epochs:
            return False
        accs = [l['dev_acc'] for l in log[-k:]]
        accs_converged = (np.max(accs) - np.min(accs)) < thresh
        if masked:
            sparsity = [l['reg_val'] for l in log[-k:]]
            sparsity_converged = (np.max(sparsity) - np.min(sparsity)) < thresh
            return accs_converged and sparsity_converged
        return accs_converged

    model = NERModel(bert_encoder)
    train_data = load_ner(train_path, model.tag2i)
    dev_data = load_ner(dev_path, model.tag2i)    

    print("lambda_init: {}, lambda_final: {}, lambda_startup_frac: {}, lambda_warmup_frac: {}".format(
        lambda_init, lambda_final, lambda_startup_frac, lambda_warmup_frac))
    print("lr_base: {}, mask_lr_base: {}, lr_warmup_frac: {}, epochs: {}, batch_size: {}".format(
        lr_base, mask_lr_base, lr_warmup_frac, epochs, batch_size))

    # group parameters for different learning rates
    mask_params = [p for n, p in model.named_parameters() if 'mask_score' in n and p.requires_grad]
    bert_params = [p for n, p in model.named_parameters() if 'mask_score' not in n and p.requires_grad]
    trainer = torch.optim.Adam([
        {'params': mask_params, 'lr': 0., 'lr_base': mask_lr_base, 'name': 'mask'},
        {'params': bert_params, 'lr': 0., 'lr_base': lr_base, 'name': 'bert'},], lr = 0.)
    
    def set_lr(lr_ratio):
        for param_group in trainer.param_groups:
            param_group['lr'] = param_group['lr_base'] * lr_ratio

    log = []
    processed = 0
    check_processed = 0
    check_every = 2048
    lambda_reg = lambda_init
    for epoch in range(epochs):#tqdm.notebook.tqdm(range(epochs)):
        np.random.shuffle(train_data)
        for i in range(0, len(train_data), batch_size):#tqdm.notebook.tqdm(range(0, len(train_data), batch_size)):
            examples = train_data[i:i+batch_size]
            model.train()
            trainer.zero_grad()
            if len(examples) == batch_size:
                # compute loss, also log other metrics
                for i in range(0, len(examples), subbatch_size):
                    examples_subbatch = examples[i:i+subbatch_size]
                    sents = [exmp['sent'] for exmp in examples]
                    labels = [exmp['labels'] for exmp in examples]
                    loss = F.cross_entropy(model.predict_batch(sents), from_numpy(pack_labels(labels)).long())
                    (loss * len(examples_subbatch) / len(examples)).backward()
                if masked:
                    reg = model.bert.compute_total_regularizer()
                    (lambda_reg * reg).backward()
                #mask_grad_norm = torch.nn.utils.clip_grad_norm_(mask_params, np.inf)
                #bert_grad_norm = torch.nn.utils.clip_grad_norm_(bert_params, np.inf)
                trainer.step()

                processed += len(examples)
                check_processed += len(examples)

                # warmup from 0 to lr_base for lr_warmup_frac
                lr_ratio = min(1, processed / (lr_warmup_frac * epochs * len(train_data)))
                set_lr(lr_ratio)
                
                # schedule lambda_reg - constant, then linear, then constant
                lambda_ratio = max(0, min(1, (processed - lambda_startup_frac * epochs * len(train_data)) / (lambda_warmup_frac * epochs * len(train_data))))
                lambda_reg = lambda_init + (lambda_final - lambda_init) * lambda_ratio

                if check_processed >= check_every:
                    if masked:
                        log.append({'dev_acc': calc_dev(),
                                    'loss_val': loss.item(), 
                                    'reg_val': reg.item(),
                                    #'mask_grad_norm': mask_grad_norm.item(), 
                                    #'bert_grad_norm': bert_grad_norm.item(), 
                                    'pct_binary': model.bert.compute_binary_pct()})
                    else:
                        log.append({'dev_acc': calc_dev(),
                                    'loss_val': loss.item()})
                    check_processed -= check_every
                    if converged(log):
                        break
                    if verbose:
                        print("Log: {}".format(log[-1]))
        if converged(log):
            break
    return log, model

def save_mask_scores(model, log, base = '../test/mask_scores'):
    keys = ['dev_acc', 'reg_val']
    fname = base
    for key in keys:
        if key in log[-1].keys():
            fname = fname + '_{}={:.4f}'.format(key, log[-1][key]*100)
    fname = fname + '.pt'

    print("Saving to {}...".format(fname))
    torch.save(partial_state_dict(model.state_dict()), fname)
    print("Done saving")
