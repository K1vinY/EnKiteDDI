
import argparse
import math
import os
import sys

sys.path.append('../KiteDDI')

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle

from build_vocab import WordVocab
from utils import split

import copy
from typing import Optional, Any

import torch
from torch import Tensor
from torch.nn.modules import Module
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
import resnet18
import random
from rdkit import Chem
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, auc, roc_curve
from sklearn.metrics import matthews_corrcoef


seed = 101
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
random.seed(seed)

PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4
SEP = 5

def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding_bert(nn.Module):
    def __init__(self, vocab_size, d_model, maxlen, n_segments):
        super(Embedding_bert, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model).cuda()
        self.pos_embed = nn.Embedding(maxlen, d_model).cuda()
        self.seg_embed = nn.Embedding(n_segments, d_model).cuda()

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).cuda()
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return embedding


class BERT1_SemMedDB(nn.Module):
    def __init__(self, vocab_size, d_model, maxlen, n_segments, n_layers, d_k, d_v, n_heads, d_ff, num_classes, semmeddb_dim=9):
        super(BERT1_SemMedDB, self).__init__()
        self.embed = Embedding_bert(vocab_size, d_model, maxlen, n_segments)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                        dim_feedforward=d_ff, dropout=0)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)

        self.resnet18 = resnet18.ResNet(img_channels=1, num_layers=18, block=resnet18.BasicBlock,
                                        num_classes=num_classes)
        
        self.self_attn2 = nn.MultiheadAttention(400, 1, dropout=0.0)
        self.norm2 = nn.LayerNorm(400)
        self.dropout2 = nn.Dropout(0.3)

        self.layer2 = nn.Sequential(nn.Linear(800,512),
                                    nn.LayerNorm(512),
                                    nn.LeakyReLU(),
                                    nn.Dropout(0.3))

        self.semmeddb_dim = semmeddb_dim
        self.semmeddb_layer = nn.Sequential(
            nn.Linear(semmeddb_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(1312 + 32, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, segment_ids, masked_pos, kg_embed, semmeddb_feat):
        embedded = self.embed(input_ids, segment_ids)
        hidden = self.encoder(embedded)
        hidden = hidden.permute(1,0,2)
        hidden1 = hidden.reshape(hidden.shape[0], 1, hidden.shape[1], hidden.shape[2])
        out = self.resnet18(hidden1)

        kg_embed = kg_embed.flatten(1)
        
        semmeddb_out = self.semmeddb_layer(semmeddb_feat)
        
        total = torch.concat((kg_embed, out, semmeddb_out), dim=1)
        logits_clsf = self.layer3(total)

        return logits_clsf


def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n_epoch', '-e', type=int, default=100, help='number of epochs')
    parser.add_argument('--num_classes', '-c', type=int, default=65, help='number of classes')
    parser.add_argument('--vocab', '-v', type=str, default='../../data/vocab_all_smiles3.pkl', help='vocabulary (.pkl)')
    parser.add_argument('--data', '-d', type=str, default='../../data/DB1_data_allFolds', help='train corpus (.csv)')
    parser.add_argument('--out-dir', '-o', type=str, default='./result', help='output directory')
    parser.add_argument('--name', '-n', type=str, default='Pretrain_Bert', help='model name')
    parser.add_argument('--seq_len', type=int, default=500, help='maximum length of the paired seqence')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size')
    parser.add_argument('--n_worker', '-w', type=int, default=8, help='number of workers')
    parser.add_argument('--hidden', type=int, default=256, help='length of hidden vector')
    parser.add_argument('--n_layer', '-l', type=int, default=6, help='number of layers')
    parser.add_argument('--n_head', type=int, default=8, help='number of attention heads')
    parser.add_argument('--lr', type=float, default=5e-5, help='Adam learning rate')
    parser.add_argument('--gpu', metavar='N', type=int, nargs='+', help='list of GPU IDs to use')
    return parser.parse_args()


def evaluate(model, test_loader):
    model.eval()
    total_loss = 0
    acc = 0
    pred_list = []
    target_list = []
    pred_score_list = []
    criterion = nn.CrossEntropyLoss()
    
    for b, d in enumerate(test_loader):
        input_ids = d[5].cuda()
        segment_ids = d[1].cuda()
        masked_pos = d[2].cuda()
        target = d[4].cuda()
        kg_embed = d[6].cuda()
        semmeddb_feat = d[7].cuda()

        with torch.no_grad():
            logits_clsf = model(torch.t(input_ids), torch.t(segment_ids), masked_pos, kg_embed, semmeddb_feat)

        loss_clsf = criterion(logits_clsf, target)
        loss = loss_clsf

        total_loss += loss.item()

        pred_score_list.extend(logits_clsf.detach().cpu().numpy())
        pred = torch.max(logits_clsf, axis=1)[1]
        pred_list.extend(pred.detach().cpu().numpy())
        target_list.extend(target.detach().cpu().numpy())
        acc += torch.sum(pred == target).item()

    f1_macro = f1_score(target_list, pred_list, average='macro')
    f1_micro = f1_score(target_list, pred_list, average='micro')
    f1_weighted = f1_score(target_list, pred_list, average='weighted')
    
    precision_macro = precision_score(target_list, pred_list, average='macro', zero_division=0)
    precision_weighted = precision_score(target_list, pred_list, average='weighted', zero_division=0)
    
    recall_macro = recall_score(target_list, pred_list, average='macro', zero_division=0)
    recall_weighted = recall_score(target_list, pred_list, average='weighted', zero_division=0)
    
    mcc = matthews_corrcoef(target_list, pred_list)
    
    try:
        from sklearn.preprocessing import label_binarize
        from scipy.special import softmax
        
        pred_score_array = np.array(pred_score_list)
        pred_prob_array = softmax(pred_score_array, axis=1)
        n_classes = pred_prob_array.shape[1]
        
        target_binarized = label_binarize(target_list, classes=range(n_classes))
        
        if len(target_binarized.shape) == 1:
            target_binarized = target_binarized.reshape(-1, 1)
        
        auc_scores = []
        aupr_scores = []
        valid_classes = 0
        
        for i in range(n_classes):
            if len(np.unique(target_binarized[:, i])) > 1:
                valid_classes += 1
                try:
                    auc_score = roc_auc_score(target_binarized[:, i], pred_prob_array[:, i])
                    auc_scores.append(auc_score)
                    
                    precision_curve, recall_curve, _ = precision_recall_curve(target_binarized[:, i], pred_prob_array[:, i])
                    aupr_score = auc(recall_curve, precision_curve)
                    aupr_scores.append(aupr_score)
                except Exception as inner_e:
                    pass
        
        auc_roc = np.mean(auc_scores) if auc_scores else 0
        aupr = np.mean(aupr_scores) if aupr_scores else 0
        
    except Exception as e:
        print(f"Warning: Failed to calculate AUC/AUPR: {e}")
        import traceback
        traceback.print_exc()
        auc_roc = 0
        aupr = 0
    
    final_loss = total_loss / len(test_loader)
    
    return (final_loss, acc/len(test_loader.dataset), 
            f1_micro, f1_macro, f1_weighted,
            precision_macro, precision_weighted,
            recall_macro, recall_weighted,
            mcc, auc_roc, aupr,
            target_list, pred_list, pred_score_list)


class Seq2seqDataset_v2(Dataset):
    def __init__(self, data, vocab, seq_len, num_classes, max_pred):
        self.vocab = vocab
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.max_pred = max_pred
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        smile1, label, smile2, embed1, embed2, name1, name2, semmeddb_feat = item
        
        sm1 = split(smile1).split()
        sm2 = split(smile2).split()
        content1_raw = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm1]
        content2_raw = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm2]
        
        if (len(content1_raw) + len(content2_raw)) > 497:
            if len(content1_raw) > 247 and len(content2_raw) > 247:
                content1 = content1_raw[0:247]
                content2 = content2_raw[0:247]
            elif len(content1_raw) > 247 and len(content2_raw) < 247:
                content1 = content1_raw[0:247]
                content2 = content2_raw
            elif len(content1_raw) < 247 and len(content2_raw) > 247:
                content2 = content2_raw[0:247]
                content1 = content1_raw
            else:
                print("Problem in Dataset")
        else:
            content1 = content1_raw
            content2 = content2_raw
        
        input_ids = [self.vocab.sos_index] + content1 + [self.vocab.sep_index] + content2 + [self.vocab.eos_index]
        input_ids_unmasked = copy.deepcopy(input_ids)
        segment_ids = [0] * (1 + len(content1) + 1) + [1] * (len(content2) + 1)
        
        n_pred = min(self.max_pred, max(1, int(round(len(input_ids) * 0.15))))
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != self.vocab.sos_index and token != self.vocab.sep_index and token != self.vocab.eos_index]
        random.shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            input_ids[pos] = self.vocab.mask_index
        
        n_pad = self.seq_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)
        input_ids_unmasked.extend([0] * n_pad)
        
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
        
        embed = torch.stack((torch.tensor(embed1), torch.tensor(embed2)), dim=0)
        
        return (
            torch.tensor(input_ids),
            torch.tensor(segment_ids),
            torch.tensor(masked_pos),
            torch.tensor(masked_tokens),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(input_ids_unmasked),
            embed,
            torch.tensor(semmeddb_feat, dtype=torch.float32)
        )


def add_kg_embed_and_semmeddb(train_data, kges_dict, db1_drugs, semmeddb_features):
    train_data_new = []
    db1_names = list(db1_drugs['name'])
    db1_smiles = list(db1_drugs['smiles'])
    
    semmeddb_feats = semmeddb_features['features']
    default_feat = semmeddb_features['default']
    
    for i in range(len(train_data)):
        smile1 = train_data[i][0]
        smile2 = train_data[i][2]
        
        name1 = db1_names[db1_smiles.index(smile1)]
        name2 = db1_names[db1_smiles.index(smile2)]
        
        embed1 = kges_dict[name1]
        embed2 = kges_dict[name2]
        
        name_pair = tuple(sorted([name1, name2]))
        semmeddb_feat = semmeddb_feats.get(name_pair, default_feat)
        
        train_data_new.append([smile1, train_data[i][1], smile2, embed1, embed2, name1, name2, semmeddb_feat])

    return train_data_new


def print_results(name, loss, acc, f1_weighted, f1_macro, mcc, 
                  precision_weighted, precision_macro,
                  recall_weighted, recall_macro, aupr, auc_val,
                  gt_list=None, pred_list=None):
    print('='*80)
    print(f'ENKITEDDI 9-dim: {name} Results:')
    print('-'*80)
    print(f'  {"Metric":<25} {"Value":>10}')
    print('-'*80)
    print(f'  {"Loss":<25} {loss:>10.4f}')
    print(f'  {"Accuracy":<25} {acc:>10.4f}')
    print(f'  {"F1 Score (Weighted)":<25} {f1_weighted:>10.4f}')
    print(f'  {"F1 Score (Macro)":<25} {f1_macro:>10.4f}')
    print(f'  {"MCC":<25} {mcc:>10.4f}')
    print(f'  {"Precision (Weighted)":<25} {precision_weighted:>10.4f}')
    print(f'  {"Precision (Macro)":<25} {precision_macro:>10.4f}')
    print(f'  {"Recall (Weighted)":<25} {recall_weighted:>10.4f}')
    print(f'  {"Recall (Macro)":<25} {recall_macro:>10.4f}')
    print(f'  {"AUPR":<25} {aupr:>10.4f}')
    print(f'  {"AUC":<25} {auc_val:>10.4f}')
    
    if gt_list is not None and pred_list is not None:
        from collections import Counter
        gt_counter = Counter(gt_list)
        pred_counter = Counter(pred_list)
        print('-'*80)
        print('  Prediction Distribution Diagnostic:')
        print(f'    Unique ground truth classes: {len(gt_counter)}')
        print(f'    Unique predicted classes: {len(pred_counter)}')
        print(f'    Top 5 predicted classes: {pred_counter.most_common(5)}')
        if len(pred_counter) <= 3:
            print(f'    WARNING: Model is only predicting {len(pred_counter)} class(es)!')
    print('='*80)


def main():

    args = parse_arguments()
    assert torch.cuda.is_available()

    print('='*80)
    print('EnKiteDDI - Ablation Study: 9-dim SemMedDB Features Evaluation')
    print('='*80)
    
    print('\nLoading dataset...')
    with open('../../data/DB1_data_allFolds', 'rb') as f:
        a = pickle.load(f)

    train_fold, valid_fold, s1_fold, s2_fold = a[0:4]
    train_data_raw = train_fold[0]
    valid_data_raw = valid_fold[0]
    s1_data_raw = s1_fold[0]
    s2_data_raw = s2_fold[0]

    kge_path = '../../data/db1_kges_transe_new.pkl'
    print(f'Loading KGE embeddings from: {kge_path}')
    with open(kge_path, 'rb') as f:
        kges_dict = pickle.load(f)

    semmeddb_path = '../../data/semmeddb_features_db1_9dim.pkl'
    print(f'Loading SemMedDB features from: {semmeddb_path}')
    with open(semmeddb_path, 'rb') as f:
        semmeddb_features = pickle.load(f)
    print(f'  SemMedDB feature pairs: {len(semmeddb_features["features"]):,}')
    print(f'  Feature dimension: {semmeddb_features["feature_dim"]}')

    db1_drugs = pd.read_csv("../../data/db1_drugs.csv")
    
    train_data = add_kg_embed_and_semmeddb(train_data_raw, kges_dict, db1_drugs, semmeddb_features)
    valid_data = add_kg_embed_and_semmeddb(valid_data_raw, kges_dict, db1_drugs, semmeddb_features)
    s1_data = add_kg_embed_and_semmeddb(s1_data_raw, kges_dict, db1_drugs, semmeddb_features)
    s2_data = add_kg_embed_and_semmeddb(s2_data_raw, kges_dict, db1_drugs, semmeddb_features)

    semmeddb_count = sum(1 for d in train_data if np.any(d[7] != 0))
    print(f'  Training pairs with SemMedDB info: {semmeddb_count} / {len(train_data)} ({semmeddb_count/len(train_data)*100:.1f}%)')

    max_pred = 65
    random.seed(101)

    vocab = WordVocab.load_vocab(args.vocab)
    dataset_valid = Seq2seqDataset_v2(valid_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes, max_pred=max_pred)
    dataset_s1 = Seq2seqDataset_v2(s1_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes, max_pred=max_pred)
    dataset_s2 = Seq2seqDataset_v2(s2_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes, max_pred=max_pred)

    valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    s1_loader = DataLoader(dataset_s1, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    s2_loader = DataLoader(dataset_s2, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)

    print(f'Evaluation size: {len(dataset_valid)}')
    print(f'U2 (Unseen Drugs) size: {len(dataset_s1)}')
    print(f'U1 (Unseen Combinations) size: {len(dataset_s2)}')

    maxlen = 500
    n_segments = 2
    d_k = d_v = 64
    d_ff = args.hidden
    torch.manual_seed(101)
    torch.cuda.manual_seed(101)
    torch.cuda.manual_seed_all(101)
    
    model_bert = BERT1_SemMedDB(
        len(vocab), args.hidden, maxlen, n_segments, 
        args.n_layer, d_k, d_v, args.n_head, d_ff, 
        args.num_classes, semmeddb_dim=9
    ).cuda()

    model_path = 'trained_models/model_enkiteddi_db1_9dim_best.pkl'
    print(f'\nLoading model from: {model_path}')
    model_bert.load_state_dict(torch.load(model_path), strict=False)

    print('\nEvaluating...\n')

    (loss_eval, acc_eval, f1_micro_eval, f1_macro_eval, f1_weighted_eval,
     precision_macro_eval, precision_weighted_eval,
     recall_macro_eval, recall_weighted_eval,
     mcc_eval, auc_eval, aupr_eval,
     gt_eval, pred_eval, score_eval) = evaluate(model_bert, valid_loader)

    print_results('Validation', loss_eval, acc_eval, f1_weighted_eval, f1_macro_eval,
                  mcc_eval, precision_weighted_eval, precision_macro_eval,
                  recall_weighted_eval, recall_macro_eval, aupr_eval, auc_eval,
                  gt_eval, pred_eval)

    (loss_U2, acc_U2, f1_micro_U2, f1_macro_U2, f1_weighted_U2,
     precision_macro_U2, precision_weighted_U2,
     recall_macro_U2, recall_weighted_U2,
     mcc_U2, auc_U2, aupr_U2,
     gt_U2, pred_U2, score_U2) = evaluate(model_bert, s1_loader)

    print_results('U2 (Unseen Drugs)', loss_U2, acc_U2, f1_weighted_U2, f1_macro_U2,
                  mcc_U2, precision_weighted_U2, precision_macro_U2,
                  recall_weighted_U2, recall_macro_U2, aupr_U2, auc_U2,
                  gt_U2, pred_U2)

    (loss_U1, acc_U1, f1_micro_U1, f1_macro_U1, f1_weighted_U1,
     precision_macro_U1, precision_weighted_U1,
     recall_macro_U1, recall_weighted_U1,
     mcc_U1, auc_U1, aupr_U1,
     gt_U1, pred_U1, score_U1) = evaluate(model_bert, s2_loader)

    print_results('U1 (Unseen Combinations)', loss_U1, acc_U1, f1_weighted_U1, f1_macro_U1,
                  mcc_U1, precision_weighted_U1, precision_macro_U1,
                  recall_weighted_U1, recall_macro_U1, aupr_U1, auc_U1,
                  gt_U1, pred_U1)

    print('\n')
    print('='*80)
    print('SUMMARY TABLE - EnKiteDDI 9-dim DB1 (Ablation Study)')
    print('='*80)
    print(f'{"Metric":<25} {"Validation":>12} {"U2":>12} {"U1":>12}')
    print('-'*80)
    print(f'{"Accuracy":<25} {acc_eval:>12.4f} {acc_U2:>12.4f} {acc_U1:>12.4f}')
    print(f'{"F1 (Weighted)":<25} {f1_weighted_eval:>12.4f} {f1_weighted_U2:>12.4f} {f1_weighted_U1:>12.4f}')
    print(f'{"F1 (Macro)":<25} {f1_macro_eval:>12.4f} {f1_macro_U2:>12.4f} {f1_macro_U1:>12.4f}')
    print(f'{"MCC":<25} {mcc_eval:>12.4f} {mcc_U2:>12.4f} {mcc_U1:>12.4f}')
    print(f'{"Precision (Weighted)":<25} {precision_weighted_eval:>12.4f} {precision_weighted_U2:>12.4f} {precision_weighted_U1:>12.4f}')
    print(f'{"Precision (Macro)":<25} {precision_macro_eval:>12.4f} {precision_macro_U2:>12.4f} {precision_macro_U1:>12.4f}')
    print(f'{"Recall (Weighted)":<25} {recall_weighted_eval:>12.4f} {recall_weighted_U2:>12.4f} {recall_weighted_U1:>12.4f}')
    print(f'{"Recall (Macro)":<25} {recall_macro_eval:>12.4f} {recall_macro_U2:>12.4f} {recall_macro_U1:>12.4f}')
    print(f'{"AUPR":<25} {aupr_eval:>12.4f} {aupr_U2:>12.4f} {aupr_U1:>12.4f}')
    print(f'{"AUC":<25} {auc_eval:>12.4f} {auc_U2:>12.4f} {auc_U1:>12.4f}')
    print('='*80)

    all_results = {
        'validation': {
            'gt': gt_eval, 'pred': pred_eval, 'score': score_eval,
            'metrics': {
                'loss': loss_eval, 'acc': acc_eval,
                'f1_weighted': f1_weighted_eval, 'f1_macro': f1_macro_eval,
                'precision_weighted': precision_weighted_eval, 'precision_macro': precision_macro_eval,
                'recall_weighted': recall_weighted_eval, 'recall_macro': recall_macro_eval,
                'mcc': mcc_eval, 'auc': auc_eval, 'aupr': aupr_eval
            }
        },
        'U2': {
            'gt': gt_U2, 'pred': pred_U2, 'score': score_U2,
            'metrics': {
                'loss': loss_U2, 'acc': acc_U2,
                'f1_weighted': f1_weighted_U2, 'f1_macro': f1_macro_U2,
                'precision_weighted': precision_weighted_U2, 'precision_macro': precision_macro_U2,
                'recall_weighted': recall_weighted_U2, 'recall_macro': recall_macro_U2,
                'mcc': mcc_U2, 'auc': auc_U2, 'aupr': aupr_U2
            }
        },
        'U1': {
            'gt': gt_U1, 'pred': pred_U1, 'score': score_U1,
            'metrics': {
                'loss': loss_U1, 'acc': acc_U1,
                'f1_weighted': f1_weighted_U1, 'f1_macro': f1_macro_U1,
                'precision_weighted': precision_weighted_U1, 'precision_macro': precision_macro_U1,
                'recall_weighted': recall_weighted_U1, 'recall_macro': recall_macro_U1,
                'mcc': mcc_U1, 'auc': auc_U1, 'aupr': aupr_U1
            }
        }
    }

    os.makedirs('./result', exist_ok=True)
    with open(r"./result/r_enkiteddi_db1_9dim.pkl", "wb") as output_file:
        pickle.dump(all_results, output_file)
    
    print(f'\nResults saved to: ./result/r_enkiteddi_db1_9dim.pkl')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)


