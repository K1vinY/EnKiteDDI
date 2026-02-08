
import argparse
import math
import os
import sys
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'KiteDDI'))

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
import resnet18
import random
from rdkit import Chem
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, auc
from sklearn.metrics import matthews_corrcoef
import copy


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
    parser.add_argument('--n_epoch', '-e', type=int, default=55, help='number of epochs')
    parser.add_argument('--num_classes', '-c', type=int, default=65, help='number of classes')
    parser.add_argument('--vocab', '-v', type=str, default='../../data/vocab_all_smiles3.pkl', help='vocabulary (.pkl)')
    parser.add_argument('--data', '-d', type=str, default='../../data/DB1_data_allFolds', help='train corpus')
    parser.add_argument('--out-dir', '-o', type=str, default='./result', help='output directory')
    parser.add_argument('--seq_len', type=int, default=500, help='maximum length of the paired seqence')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size')
    parser.add_argument('--n_worker', '-w', type=int, default=8, help='number of workers')
    parser.add_argument('--hidden', type=int, default=256, help='length of hidden vector')
    parser.add_argument('--n_layer', '-l', type=int, default=6, help='number of layers')
    parser.add_argument('--n_head', type=int, default=8, help='number of attention heads')
    parser.add_argument('--lr', type=float, default=5e-5, help='Adam learning rate')
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
        total_loss += loss_clsf.item()

        pred_score_list.extend(logits_clsf.detach().cpu().numpy())
        pred = torch.max(logits_clsf, axis=1)[1]
        pred_list.extend(pred.detach().cpu().numpy())
        target_list.extend(target.detach().cpu().numpy())
        acc += torch.sum(pred == target).item()

    f1_macro = f1_score(target_list, pred_list, average='macro')
    f1_micro = f1_score(target_list, pred_list, average='micro')
    f1_avg = f1_score(target_list, pred_list, average='weighted')
    f1_bin = matthews_corrcoef(target_list, pred_list)
    
    final_loss = total_loss / len(test_loader)
    return final_loss, acc/len(test_loader.dataset), f1_micro, f1_macro, f1_avg, f1_bin, target_list, pred_list, pred_score_list


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


def main():
    args = parse_arguments()
    
    if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] != '0':
        print(f'WARNING: CUDA_VISIBLE_DEVICES was set to {os.environ["CUDA_VISIBLE_DEVICES"]}, forcing to 0')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.get_device_name(0)}')
        torch.cuda.set_device(0)

    print('='*70)
    print('EnKiteDDI - Ablation Study: 9-dim SemMedDB Features')
    print('='*70)
    
    print('\nLoading dataset...')
    with open('../../data/DB1_data_allFolds', 'rb') as f:
        a = pickle.load(f)

    train_fold, valid_fold, s1_fold, s2_fold = a[0:4]
    train_data_raw = train_fold[0]
    valid_data_raw = valid_fold[0]
    s1_data_raw = s1_fold[0]
    s2_data_raw = s2_fold[0]

    print('Loading ORIGINAL KG embeddings: ../../data/db1_kges_transe_new.pkl')
    with open('../../data/db1_kges_transe_new.pkl', 'rb') as f:
        kges_dict = pickle.load(f)

    print('Loading SemMedDB features: ../../data/semmeddb_features_db1_9dim.pkl')
    with open('../../data/semmeddb_features_db1_9dim.pkl', 'rb') as f:
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
    dataset_train = Seq2seqDataset_v2(train_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes, max_pred=max_pred)
    dataset_valid = Seq2seqDataset_v2(valid_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes, max_pred=max_pred)
    dataset_s1 = Seq2seqDataset_v2(s1_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes, max_pred=max_pred)
    dataset_s2 = Seq2seqDataset_v2(s2_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes, max_pred=max_pred)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    test_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    s1_loader = DataLoader(dataset_s1, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    s2_loader = DataLoader(dataset_s2, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)

    print(f'Train size: {len(dataset_train)}')
    print(f'Test size: {len(dataset_valid)}')
    print(f's1 (U2) size: {len(dataset_s1)}')
    print(f's2 (U1) size: {len(dataset_s2)}')

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

    pretrain_path = "../KiteDDI/model/bert_pretrain_vocab7_793_2704.pkl"
    if os.path.exists(pretrain_path):
        model_bert.load_state_dict(torch.load(pretrain_path), strict=False)
        print(f"Loaded pretrained model: {pretrain_path}")
    else:
        print(f"WARNING: Pretrained model not found: {pretrain_path}")

    optimizer_bert = optim.Adam(model_bert.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion_bert = nn.CrossEntropyLoss()

    best_acc = 0
    best_epoch = 0
    all_results = []
    
    for e in range(1, args.n_epoch):
        print(f"\n>>> Epoch: {e}")
        model_bert.train()
        
        for b, d in tqdm(enumerate(train_loader)):
            input_ids = d[5].cuda()
            segment_ids = d[1].cuda()
            masked_pos = d[2].cuda()
            target = d[4].cuda()
            kg_embed = d[6].cuda()
            semmeddb_feat = d[7].cuda()

            optimizer_bert.zero_grad()
            logits_clsf = model_bert(torch.t(input_ids), torch.t(segment_ids), masked_pos, kg_embed, semmeddb_feat)
            loss_clsf = criterion_bert(logits_clsf, target)
            loss_clsf.backward()
            optimizer_bert.step()
            
            if b % 100 == 0:
                print(f'Train {e}: iter {b} | loss {loss_clsf.item():.4f}')

        loss_train, acc_train, f1_micro, f1_macro, f1_avg, f1_bin, gt_train, pred_train, score_train = evaluate(model_bert, train_loader)
        print(f'Train {e}: loss {loss_train:.4f} | acc {acc_train:.4f} | f1_macro {f1_macro:.4f} | MCC {f1_bin:.4f}')

        loss_val, acc_val, f1_micro1, f1_macro1, f1_avg1, f1_bin1, gt_eval, pred_eval, score_eval = evaluate(model_bert, test_loader)
        print(f'Val {e}: loss {loss_val:.4f} | acc {acc_val:.4f} | f1_macro {f1_macro1:.4f} | MCC {f1_bin1:.4f}')

        loss_U2, acc_U2, f1_micro2, f1_macro2, f1_avg2, f1_bin2, gt_U2, pred_U2, score_U2 = evaluate(model_bert, s1_loader)
        print(f'U2 {e}: loss {loss_U2:.4f} | acc {acc_U2:.4f} | f1_macro {f1_macro2:.4f} | MCC {f1_bin2:.4f}')

        loss_U1, acc_U1, f1_micro3, f1_macro3, f1_avg3, f1_bin3, gt_U1, pred_U1, score_U1 = evaluate(model_bert, s2_loader)
        print(f'U1 {e}: loss {loss_U1:.4f} | acc {acc_U1:.4f} | f1_macro {f1_macro3:.4f} | MCC {f1_bin3:.4f}')

        if acc_U2 > best_acc:
            best_acc = acc_U2
            best_epoch = e
            os.makedirs('./trained_models', exist_ok=True)
            torch.save(model_bert.state_dict(), 'trained_models/model_enkiteddi_db1_9dim_best.pkl')
            print(f'\n*** Best model saved! Epoch {e}, U2 acc: {acc_U2:.4f} ***\n')
        
        if e % 10 == 0:
            os.makedirs('./trained_models', exist_ok=True)
            torch.save(model_bert.state_dict(), f'trained_models/model_enkiteddi_db1_9dim_epoch{e}.pkl')

        all_results.append([gt_train, pred_train, score_train, gt_eval, pred_eval, score_eval, gt_U2, pred_U2, score_U2, gt_U1, pred_U1, score_U1])

    os.makedirs('./trained_models', exist_ok=True)
    torch.save(model_bert.state_dict(), 'trained_models/model_enkiteddi_db1_9dim_final.pkl')
    
    print('\n' + '='*70)
    print('EnKiteDDI 9-dim Training Completed!')
    print('='*70)
    print(f'Best Epoch: {best_epoch} | Best U2 Accuracy: {best_acc:.4f}')
    print(f'Final model: trained_models/model_enkiteddi_db1_9dim_final.pkl')
    print(f'Best model: trained_models/model_enkiteddi_db1_9dim_best.pkl')
    print('='*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


