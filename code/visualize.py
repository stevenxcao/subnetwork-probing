import numpy as np
from matplotlib import pyplot as plt
import torch
from operator import attrgetter

def visualize_head_sparsity(model, path, block = False, layers = 12, heads = 12): 
    """
    bert_lm.bert.encoder.layer.0.attention.self.query.mask_scores
    percentage of each head in each model that is on, displayed as a grid
    """
    plt.rcParams['figure.dpi'] = 100
    sparsity_vals = np.zeros((layers, heads)) # (y, x)
    for layer in range(layers):
        for head in range(heads):
            q, k, v = model.get_sparsity(layer, head)
            sparsity_vals[layer, head] = (q + k + v) / 3
    fig, ax = plt.subplots()
    ax.imshow(sparsity_vals) 
    for (j,i),label in np.ndenumerate(sparsity_vals):
        if label > np.max(sparsity_vals) / 2:
            color = 'black'
        else:
            color = 'white'
        ax.text(i, j, int(label * 10000) / 100, ha='center', va='center', size = 'x-small', color = color)
    # We want to show all ticks...
    ax.set_xticks(np.arange(heads))
    ax.set_yticks(np.arange(layers))
    ax.set_title("Sparsity level of each head in each layer")
    plt.xlabel('Heads')
    plt.ylabel('Layers')
    fig.tight_layout()
    if block:
        plt.show()
    else:
        plt.savefig('{}_{}_graph.png'.format(path, 'head_sparsity'))
        plt.show(block=False)
        plt.close()
    
def visualize_dense_sparsity(model, path, block = False, layers = 12): 
    """
    bert_lm.bert.encoder.layer.0.attention.self.query.mask_scores
    percentage of each head in each model that is on, displayed as a grid
    """
    plt.rcParams['figure.dpi'] = 100
    sparsity_vals = np.zeros((layers, 3)) # (y, x)
    for layer in range(layers):
        a, b, c = model.get_sparsity_dense(layer)
        sparsity_vals[layer, 0] = a
        sparsity_vals[layer, 1] = b
        sparsity_vals[layer, 2] = c
    fig, ax = plt.subplots()
    ax.imshow(sparsity_vals) 
    for (j,i),label in np.ndenumerate(sparsity_vals):
        if label > np.max(sparsity_vals) / 2:
            color = 'black'
        else:
            color = 'white'
        ax.text(i, j, int(label * 10000) / 100, ha='center', va='center', size = 'x-small', color = color)
    # We want to show all ticks...
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(layers))
    ax.set_title("Sparsity level of each dense layer")
    plt.xlabel('Dense1/Dense2/Dense3')
    plt.ylabel('Layers')
    fig.tight_layout()
    if block:
        plt.show()
    else:
        plt.savefig('{}_{}_graph.png'.format(path, 'dense_sparsity'))
        plt.show(block=False)
        plt.close()

def visualize_layer_attn_sparsity(model, path, block = False, layers = 12, normalize = True): 
    """
    bert_lm.bert.encoder.layer.0.attention.self.query.mask_scores
    percentage of each head in each model that is on, displayed as a grid
    """
    plt.rcParams['figure.dpi'] = 100

    axes = plt.gca()
    axes.set_ylim([0,1])
    sparsities = np.array([model.get_sparsity_layer_attn(l).item() for l in range(layers)])
    if normalize:
        sparsities = sparsities / np.sum(sparsities)
    plt.bar(np.arange(layers), sparsities)

    plt.title("Sparsity level of each layer")
    plt.xlabel('Layer')
    plt.ylabel('Sparsity Level')
    if block:
        plt.show()
    else:
        plt.savefig('{}_{}_graph.png'.format(path, 'layer_attn_sparsity'))
        plt.show(block=False)
        plt.close()
