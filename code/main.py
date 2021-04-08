import sys

from bert import WordLevelBert
from masked_bert import MaskedWordLevelBert
from classifiers import BertEncoder
from train import train_ner, train_ud, train_pos, save_mask_scores
from visualize import visualize_head_sparsity, visualize_dense_sparsity, visualize_layer_attn_sparsity

from matplotlib import pyplot as plt
plt.switch_backend('agg')
import numpy as np

mask_lr_base = 0.2
lr_base = 5e-05
lambda_init = 0
lambda_final = 1
verbose = True
base_path = sys.argv[1]
expt_name = 'expt_1'

for task in ('NER', 'UD', 'UPOS'):
    for setting in ('pretrained', 'resetenc', 'resetall'):
        for method in ('prune', 'mlp1'):
            if method == 'mlp1':
                params_list = (1, 2, 5, 10, 25, 50, 125, 250, 768)
            elif method == 'finetune':
                params_list = (1,)
            elif method == 'prune':
                params_list = ((768,768), (768,192), (768,24), (768,6), 
                               (768,1), (192,1), (24,1), (6,1), (1,1))
                
            for params in params_list:
                epochs = 20
                print(task)
                print(method)
                if method == 'mlp1':
                    rank = params
                    print("MLP1 - Rank: {}".format(rank))
                    bert = WordLevelBert('bert-base-uncased')
                    bert.freeze_bert()
                    bert_encoder = BertEncoder(bert, mlp1 = True, rank = rank)
                    masked = False
                elif method == 'finetune':
                    print("Fine-tuning")
                    bert = WordLevelBert('bert-base-uncased')
                    bert_encoder = BertEncoder(bert, mlp1 = False)
                    masked = False
                elif method == 'prune':
                    out_w_per_mask, in_w_per_mask = params
                    print("Prune - (out,in)_w_per_mask: {}".format((out_w_per_mask, in_w_per_mask)))
                    bert = MaskedWordLevelBert('bert-base-uncased', out_w_per_mask, in_w_per_mask)
                    bert.freeze_bert()
                    bert_encoder = BertEncoder(bert, mlp1 = False)
                    masked = True

                if setting == 'pretrained':
                    print("Keeping pre-trained")
                elif setting == 'resetenc':
                    print("Resetting encoder!")
                    bert.reset_weights(encoder_only = True)
                elif setting == 'resetall':
                    print("Resetting all!")
                    bert.reset_weights(encoder_only = False)
                
                kwargs = {'lambda_init' : lambda_init, 'lambda_final' : lambda_final, 
                          'epochs' : epochs, 'lr_base' : lr_base, 'mask_lr_base' : mask_lr_base, 
                          'verbose' : verbose, 'masked' : masked}
                print(kwargs)

                print("Finding subnetwork...")
                if task == "NER":
                    log, model = train_ner(bert_encoder, '../data/CoNLL-NER/eng.train', 
                                           '../data/CoNLL-NER/eng.testa', **kwargs)
                elif task == "UD":
                    log, model = train_ud(bert_encoder, '../data/UD_English/en-ud-train.conllu', 
                                          '../data/UD_English/en-ud-dev.conllu', **kwargs)
                elif task == "UPOS":
                    log, model = train_pos(bert_encoder, '../data/UD_English/en-ud-train.conllu', 
                                           '../data/UD_English/en-ud-dev.conllu', **kwargs)
                
                path = '{}/{}_{}_setting={}_method={}_params={}'.format(
                    base_path, expt_name, task, setting, method, str(params).replace(' ',''))
                save_mask_scores(model, log, base = path)

                print("Final results: {}".format(log[-1]))
                for key in log[0].keys():
                    plt.plot(np.arange(len(log)), [a[key] for a in log], color='blue')
                    plt.title(key)
                    plt.savefig('{}_{}_graph.png'.format(path, key))
                    plt.show(block=False)
                    plt.close()

                if masked:
                    visualize_head_sparsity(bert, path, block = False)
                    visualize_dense_sparsity(bert, path, block = False)
                    visualize_layer_attn_sparsity(bert, path, block = False)
                    print("Percentage of elements within 0.01 of 0 or 1: {:.5f}".format(bert.compute_binary_pct()))
