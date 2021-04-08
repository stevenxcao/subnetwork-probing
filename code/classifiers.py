import torch
import torch.nn as nn
import torch.nn.init as init

from util import use_cuda

class BertEncoder(nn.Module):
    def __init__(self, bert, dropout_p = 0.1, mlp1 = False, rank = None):
        super().__init__()
        self.bert = bert
        d_features = self.bert.dim
        self.dropout = nn.Dropout(p = 0.1)
        self.use_mlp1 = mlp1
        if mlp1:
            if rank is None:
                rank = d_features
            self.mlp1 = nn.Sequential(
                    nn.Linear(d_features, rank),
                    nn.Linear(rank, d_features),
                    nn.LayerNorm(d_features),
                    nn.ReLU())
                
    def forward(self, sentences, **kwargs):
        ann, pad_mask = self.bert(sentences, **kwargs)
        ann = self.dropout(ann)
        if self.use_mlp1:
            ann = self.mlp1(ann)
        return ann, pad_mask

class POSModel(nn.Module):
    def __init__(self, bert_encoder, pos_vocab_size):
        super().__init__()
        self.bert_encoder = bert_encoder
        self.bert = self.bert_encoder.bert
        d_features = self.bert_encoder.bert.dim
        self.pos_tip = nn.Linear(d_features, pos_vocab_size)
        
        if use_cuda:
            self.cuda()
    
    def predict_batch(self, sentences):
        ann, pad_mask = self.bert_encoder(sentences)
        pos_pred = self.pos_tip(ann) # num_sent x pad_len x pos_vocab_size 
        return pos_pred

class UDModel(nn.Module):
    def __init__(self, bert_encoder, pos_vocab_size, arc_vocab_size):
        super().__init__()
        self.bert_encoder = bert_encoder
        self.bert = self.bert_encoder.bert
        d_features = self.bert_encoder.bert.dim
        
        d_label_hidden = 1024
        d_label_hidden_arc = 256

        self.mlp_head = nn.Sequential(
            nn.Linear(d_features, d_label_hidden), # 1024
            nn.ReLU(),
            nn.Dropout(p = 0.33)
            )
        self.mlp_dep = nn.Sequential(
            nn.Linear(d_features, d_label_hidden), # 1024
            nn.ReLU(),
            nn.Dropout(p = 0.33)
            )
        self.W = nn.Linear(d_label_hidden, d_label_hidden)
        self.head_vec = nn.Parameter(torch.Tensor(d_label_hidden))
        self.dep_vec = nn.Parameter(torch.Tensor(d_label_hidden))
        self.bias = nn.Parameter(torch.Tensor(1, 1, 1))
        
        init.normal_(self.head_vec)
        init.normal_(self.dep_vec)
        init.normal_(self.bias)
        
        self.mlp_head_arc = nn.Sequential(
            nn.Linear(d_features, d_label_hidden_arc), # 256
            nn.ReLU(),
            nn.Dropout(p = 0.33)
            )
        self.mlp_dep_arc = nn.Sequential(
            nn.Linear(d_features, d_label_hidden_arc), # 256
            nn.ReLU(),
            nn.Dropout(p = 0.33)
            )
        self.W_arc = nn.Parameter(torch.Tensor(arc_vocab_size,
                   d_label_hidden_arc, d_label_hidden_arc))
        self.head_vec_arc = nn.Parameter(torch.Tensor(
                d_label_hidden_arc, arc_vocab_size))
        self.dep_vec_arc = nn.Parameter(torch.Tensor(
                d_label_hidden_arc, arc_vocab_size))
        self.bias_arc = nn.Parameter(torch.Tensor(1, 1, arc_vocab_size))
        
        init.normal_(self.W_arc)
        init.normal_(self.head_vec_arc)
        init.normal_(self.dep_vec_arc)
        init.normal_(self.bias_arc)
        
        self.arc_vocab_size = arc_vocab_size
        
        self.pos_tip = nn.Linear(d_features, pos_vocab_size)
        
        if use_cuda:
            self.cuda()
    
    def predict_batch(self, sentences, correct_heads):
        # correct_heads : num_sent x pad_len, with index of correct head or -1
        # indexing starts at 1, which works out because we insert cls at 0
        # num_sent x pad_len x bert_dim, num_sent x pad_len
        ann, pad_mask = self.bert_encoder(sentences)

        pos_pred = self.pos_tip(ann) # num_sent x pad_len x pos_vocab_size
        
        ###### Arc prediction
        # [CLS] -> the root of the sentence
        # prediction:
        # num_sent x pad_len x pad_len
        # pred[n, i, j]: for sent n, dep i, head j
        # softmax over the last dimension (each dep has 1 head)
        ann_head = self.mlp_head(ann) # num_sent x pad_len x 1024
        ann_dep = self.mlp_dep(ann) # num_sent x pad_len x 1024
        #print("Initial", ann_head.shape, ann_dep.shape)
        
        head_bias = ann_head @ self.head_vec
        dep_bias = ann_dep @ self.dep_vec
        #print("Biases", head_bias.shape, dep_bias.shape)
        
        ann_head = self.W(ann_head).transpose(-1,-2) 
        #print("Post W", ann_head.shape)
        dep_W_headT = torch.bmm(ann_dep, ann_head) # num_sent x pad_len x pad_len
        #print("Post depWhead", dep_W_headT.shape)
        
        # num_sent x 1 x pad_len
        dep_W_headT += head_bias.unsqueeze(-2)
        # num_sent x pad_len x 1
        dep_W_headT += dep_bias.unsqueeze(-1) # add separately for broadcasting!
        dep_W_headT += self.bias
        
        pad_mask = 1 - pad_mask.unsqueeze(-2) # num_sent x 1 x pad_len
        #print("Padmask", pad_mask.shape)
        dep_W_headT.data.masked_fill_(pad_mask.byte(), -float('inf'))
        
        ###### Arc label prediction
        # label prediction: for arc from head-> dep, label with type
        # num_sent x pad_len (idx for dep) x label_vocab_size
        ann_head_arc = self.mlp_head_arc(ann) # num_sent x pad_len x 256
        ann_dep_arc = self.mlp_dep_arc(ann) # num_sent x pad_len x 256
        #print("Initial", ann_head_arc.shape, ann_dep_arc.shape)
        
        # select ground truth head for each dep
        ann_head_arc_perdep = None
        num_sent, pad_len, dim = ann_head_arc.shape
        for i in range(num_sent):
            ann_head_arc_perdep_sent = None
            for j in range(pad_len):
                if correct_heads[i,j] == -1:
                    a_head = torch.zeros(dim, device = ann_dep_arc.device)
                else:
                    a_head = ann_head_arc[i,correct_heads[i,j]]
                a_head = a_head.unsqueeze(0)
                if ann_head_arc_perdep_sent is None:
                    ann_head_arc_perdep_sent = a_head
                else:
                    ann_head_arc_perdep_sent = torch.cat(
                            (ann_head_arc_perdep_sent, a_head), dim = 0)
            ann_head_arc_perdep_sent = ann_head_arc_perdep_sent.unsqueeze(0)
            if ann_head_arc_perdep is None:
                ann_head_arc_perdep = ann_head_arc_perdep_sent
            else:
                ann_head_arc_perdep = torch.cat((ann_head_arc_perdep,
                                        ann_head_arc_perdep_sent), dim = 0)
        assert ann_head_arc_perdep.shape == ann_dep_arc.shape
        ann_head_arc = ann_head_arc_perdep
        
        # num_sent x pad_len x label_vocab_size
        head_arc_bias = ann_head_arc @ self.head_vec_arc
        dep_arc_bias = ann_dep_arc @ self.dep_vec_arc
        #print("Biases", head_arc_bias.shape, dep_arc_bias.shape)
        
        # head - num_sent x pad_len x 256 -> vocab x num_sent x pad_len x 256
        # -> num_sent x pad_len x vocab x 256
        # dep - num_sent x pad_len x 256 x 1
        # vocab x num_sent x pad_len x 256
        ann_head_arc = ann_head_arc.unsqueeze(0).repeat(self.arc_vocab_size,
                                             1,1,1)
        vocab = self.arc_vocab_size
        ann_head_arc = ann_head_arc.reshape(vocab, num_sent * pad_len, dim)
        # (num_sent * pad_len) x vocab x 256
        ann_head_arc = torch.bmm(ann_head_arc, self.W_arc).reshape(
                vocab, num_sent, pad_len, dim).permute(1,2,0,3).reshape(
                        num_sent * pad_len, vocab, dim)
        #print("Head post-W", ann_head_arc.shape)
        
        # (num_sent * pad_len) x 256 x 1
        ann_dep_arc = ann_dep_arc.reshape(num_sent * pad_len, dim).unsqueeze(-1)
        #print("Dep, about to hit head", ann_dep_arc.shape)
        # (num_sent * pad_len) x vocab x 1
        dep_W_headT_arc = torch.bmm(ann_head_arc, ann_dep_arc)
        dep_W_headT_arc = torch.squeeze(dep_W_headT_arc).reshape(
                num_sent, pad_len, vocab)
        #print("DepWhead_arc", dep_W_headT_arc.shape)
        # num_sent x pad_len x vocab
        dep_W_headT_arc += head_arc_bias + dep_arc_bias + self.bias_arc
        
        # pos_pred: num_sent x pad_len x pos_vocab_size
        # dep_W_headT: num_sent x pad_len x pad_len
        # dep_W_headT_arc: num_sent x pad_len x label_vocab_size
        return pos_pred, dep_W_headT, dep_W_headT_arc

class NERModel(nn.Module):
    def __init__(self, bert_encoder):
        super().__init__()
        self.tags = ('O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC')
        self.i2tag = {i:s for i, s in enumerate(self.tags)}
        self.tag2i = {s:i for i, s in enumerate(self.tags)}

        self.bert_encoder = bert_encoder
        self.bert = self.bert_encoder.bert
        d_features = self.bert_encoder.bert.dim
        self.span_tip = nn.Linear(d_features, len(self.tags))
        
        if use_cuda:
            self.cuda()
    
    def predict_batch(self, sentences):
        ann1, _ = self.bert_encoder(sentences, padded = False, include_clssep = False)
        return self.span_tip(ann1)
