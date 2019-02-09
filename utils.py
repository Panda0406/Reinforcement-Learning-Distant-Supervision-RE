# -*- coding: utf-8 -*-
from __future__ import division
import sys, os
import random
import numpy as np
from copy import deepcopy
from Networks import Rank_CNN
import datetime
import cPickle as pickle
from collections import defaultdict

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

def generate_trainloader(inputs, x, y, batch_size, shuf=True):
    x = inputs[x]
    train_data = Data.TensorDataset(torch.LongTensor(x), torch.LongTensor(y))
    trainloader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=shuf)
    return trainloader

def calculate_logits(model, x, max_sent, rv_matrix=None):
    s, p1, p2 = torch.split(x, max_sent, dim=1)
    s, p1, p2 = Variable(s), Variable(p1), Variable(p2)
    s, p1, p2 = s.type(LongTensor), p1.type(LongTensor), p2.type(LongTensor)
    if type(rv_matrix) == type(None):
        logits = model(s, p1, p2)
        return logits
    else:
        logits, sent_vecs = model(s, p1, p2, rv_matrix)
        return logits, sent_vecs

def evaluate(model, x, max_sent, rv_matrix=None):
    model.eval()
    return calculate_logits(model, x, max_sent, rv_matrix=rv_matrix)

def calculate_loss(logits, y, weights=[1.0, 1.0]):
    weights = Variable(torch.cuda.FloatTensor(weights))
    y = Variable(y).type(LongTensor)
    loss_ = F.cross_entropy(logits, y, weights)
    return loss_

def calculate_prob(model, x, y, max_sent, inputs, rv=None):
    model.eval()
    all_probs = list()
    trainloader = generate_trainloader(inputs, x, y, 1000, shuf=False)
    for i, (x,_) in enumerate(trainloader, 0):
        if type(rv) == type(None):
            logits = calculate_logits(model, x, max_sent)
        else:
            rv_matrix = rv.repeat(x.size(0), 1)
            logits, _ = calculate_logits(model, x, max_sent, rv_matrix=rv_matrix)
        probs = F.softmax(logits, dim=1).view(len(x), -1)
        probs = list(probs.data.type(torch.FloatTensor).numpy())
        all_probs += probs
    return np.array(all_probs)

def calculate_accuracy(trainloader, model, args, epoch, Policy=False):
    # Test
    model.eval()
    corrects, total, accuracy = 0, 0, 0
    corrects_non_NA, total_non_NA, accuracy_non_NA = 0, 0, 0
    for i, (x, y) in enumerate(trainloader, 0):
        if Policy:
            rv = torch.zeros(1,100)
            rv = Variable(rv.type(FloatTensor))
            rv_matrix = rv.repeat(x.size(0), 1)
            eval_logits, _ = evaluate(model, x, args.max_sent, rv_matrix=rv_matrix)
        else:
            eval_logits = evaluate(model, x, args.max_sent)
        y = y.type(LongTensor)
        corrects += (torch.max(eval_logits, 1)[1].view(y.size()).data == y).sum()
        total += int(y.size()[0])

        non_NA_index = np.where(y.type(torch.LongTensor).numpy() > 0)[0]

        if len(non_NA_index) != 0:
            corrects_non_NA += (torch.max(eval_logits, 1)[1].view(y.size()).data[non_NA_index] == y[non_NA_index]).sum()
        total_non_NA += int((y > 0).sum())
    accuracy = 100.0 * corrects / total
    accuracy_non_NA = 100.0 * corrects_non_NA / total_non_NA
    print "[Policy Pretrain] epoch: %d, acc: %f, acc_non_NA: %f = (%d/%d)" %(epoch, accuracy, accuracy_non_NA, corrects_non_NA, total_non_NA)
    return accuracy, accuracy_non_NA

def calculate_F1(model, x, y, inputs, max_sent):
    model.eval()
    correct_pos, correct_neg = 0, 0
    total_pos, total_neg = 1e-06, 1e-06
    true_pos, true_neg = 1e-06, 1e-06
    testloader = generate_trainloader(inputs, x, y, 1000, shuf=False)
    for i, (x,y) in enumerate(testloader, 0):
        logits = evaluate(model, x, max_sent)
        y = y.type(LongTensor)
        y_pred = torch.max(logits.data, 1)[1]
        total_pos += y_pred.sum()
        total_neg += y.size(0) - y_pred.sum()
        for num in range(len(y_pred)):
            if y_pred[num] == 1 and  y[num] == 1:
                correct_pos += 1
            if y_pred[num] == 0 and  y[num] == 0:
                correct_neg += 1
        true_pos += y.sum()
        true_neg += y.size(0) - y.sum()

    accuracy_pos, recall_pos = correct_pos/total_pos, correct_pos/true_pos
    F1_pos = 2*accuracy_pos*recall_pos/(accuracy_pos+recall_pos + 1e-06)
    accuracy_neg, recall_neg = correct_neg/total_neg, correct_neg/true_neg
    F1_neg = 2*accuracy_neg*recall_neg/(accuracy_neg+recall_neg + 1e-06)
    F1 = (F1_pos+F1_neg)/2
    return F1

def rank_by_prob(model, sents, inputs, max_sent, rv=None, label=1):
    sent2prob, pos_probs = dict(), list()
    x, y = sents, [label]*len(sents)
    # calculate_prob(model, x, y, max_sent, inputs, rv=None)
    pos_probs = calculate_prob(model, x, y, max_sent, inputs, rv=rv)
    pos_probs = list(pos_probs[:,label])
    assert len(pos_probs) == len(sents)
    sent2prob = dict(zip(sents, pos_probs))
    sent2prob_ranked = sorted(sent2prob.items(), key=lambda d:d[1], reverse = True)  # [(s1, p1), ...]

    return sent2prob_ranked

def select_negative(model, sents, inputs, max_sent):
    neg_ranked = rank_by_prob(model, sents, inputs, max_sent)
    return [item[0] for item in neg_ranked[:300000]]


def filter_negative(embeddings, inputs, rela2sents, args):
    """
    Select high-quality negatve samples for robust result.
    """
    print '\n\n############## Training Rank_CNN ##############'

    pos_data, neg_data = list(), list()
    for k, v in rela2sents.items():
        if k != 0:
            pos_data += v
    neg_data = rela2sents[0]

    print 'positive size:', len(pos_data), 'negative size:', len(neg_data)

    rank_model = Rank_CNN(embeddings, args, emb_change=True)
    rank_model = rank_model.cuda() if use_cuda else rank_model

    x = pos_data+neg_data
    y = [1]*len(pos_data)+[0]*len(neg_data)
    trainloader = generate_trainloader(inputs, x, y, args.batch_size, shuf=True)
    parameters = filter(lambda p: p.requires_grad, rank_model.parameters())
    optimizer = optim.Adam(parameters, lr=0.0005)

    for epoch in range(10):
        #rank_model.train()
        corrects, total, accuracy = 0, 0, 0
        for i, (x, y) in enumerate(trainloader, 0):
            logits = calculate_logits(rank_model, x, args.max_sent)
            y = y.type(LongTensor)
            loss = calculate_loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            corrects += (torch.max(logits, 1)[1].view(y.size()).data == y).sum()
            total += int(y.size()[0])

            if i % args.log_interval == 0:
                accuracy = 100.0 * corrects / total
                time_str = datetime.datetime.now().isoformat()
                sys.stdout.write("[rank_data] epoch %d step %d | loss : %f, accuracy: %f" % (epoch, i, loss.data[0], accuracy) + '\r')
                sys.stdout.flush()
        print ''

        accuracy, accuracy_non_NA = calculate_accuracy(trainloader, rank_model, args, epoch)

        if accuracy_non_NA > 90.0:
            break

    torch.save(rank_model.state_dict(), './models/Rank.pkl')
    RL_train_data = deepcopy(rela2sents)

    print "\n##############  Selecting high-quality negative samples  ##############"
    good_negative = select_negative(rank_model, rela2sents[0], inputs, args.max_sent)
    RL_train_data[0] = good_negative
    print "We select %d high-quality negative sentences" % (len(good_negative))
    pickle.dump(RL_train_data, open('./data/RL_train_data.pkl', 'wb'))

    return RL_train_data

def split_data(temp_sort, den=3):
    train_list, test_list = list(), list()
    for num in range(len(temp_sort)):
        if num%den in [0,1]:
            train_list.append(temp_sort[num])
        elif num%den == 2:
            test_list.append(temp_sort[num])
    return train_list, test_list

def generate_new_dataset(sent2sents, instances, false_positive, repetitive_tripples, tripple2sents, rela2tripple, args, fname="train.txt", train_version=522611):
    rs_num, fpt_num, fps_num = 0, 0, 0

    rela_remove_tripple_num = defaultdict(float)

    fp_sents = list()
    for sent_id in false_positive:
        rs_num += len(sent2sents[sent_id])
        fp_sents += sent2sents[sent_id]
    fp_tripple2sents = defaultdict(list)
    for sent_id in fp_sents:
        instance = instances[sent_id]
        tripple = instance['e1_id'] + "\t" +instance['e2_id'] + "\t" + str(instance['y_id'])
        fp_tripple2sents[tripple].append(sent_id)

    # remove the tripples whose sentences are all false positive.
    case57 = 0
    for tripple, sents in fp_tripple2sents.items():
        if set(sents) == set(tripple2sents[tripple]):
            if train_version == 570088 and repetitive_tripples.has_key(tripple):
                case57 += 1
                continue
            rela = tripple.split('\t')[2]
            rela2tripple[rela].remove(tripple)
            rela2tripple['0'].add(tripple)
            rela_remove_tripple_num[rela] += 1
            fpt_num += 1
            fps_num += len(sents)
            for sent in sents:
                instances[sent]['y'] = 'NA'
                instances[sent]['y_id'] = 0
    #print "Case 570088:", case57
    print "We find %d false postive sentences, and there are %d tripples in it. We remove %d tripples and %d sentences." % (rs_num, len(fp_tripple2sents), fpt_num, fps_num)

    all_tripples = tripple2sents.keys()
    print 'all_tripples_ori:', len(all_tripples)
    for rela in rela_remove_tripple_num.keys():
        tripples = rela2tripple[rela]
        add_list = list()
        for tripple in tripples:
            if len(tripple2sents[tripple]) > 10:
                add_list.append(tripple + '_add')
        #print "RELA", rela, "REMOVE_NUM", rela_remove_tripple_num[rela], "HAS_TRIPPLE", len(rela2tripple[rela]), "ADD_LIST", len(add_list)
        all_tripples += add_list

    #print 'all_tripples_new:', len(all_tripples)
    random.shuffle(all_tripples)

    if not os.path.exists(args.cleaned_data_dir):
        os.mkdir(args.cleaned_data_dir)
    ftrain = open(args.cleaned_data_dir + fname, 'w')

    rp_remove = 0
    tpnew = 0
    fp_sents = set(fp_sents)
    for tripple in all_tripples:
        if train_version == 522611 and repetitive_tripples.has_key(tripple):
            continue
        if tripple.endswith('_add'):
            tpnew += 1
            sents_ = tripple2sents[tripple[:-4]]
            sents_ = random.sample(sents_, int(len(sents_)/2.0))
            sents_set = set(sents_)
            if sents_set.issubset(fp_sents):
                #print "SUBSET!!", len(sents_set)
                continue

        else:
            sents_ = tripple2sents[tripple]
        for idx in sents_:
            s = instances[idx]
            sentence = s['text']
            if tripple[-4:] == '_add':
                line = s['e1_id'] + '_add' + '\t' + s['e2_id'] + "_add" + '\t' + s['e1'] + '\t' + s['e2'] + '\t' + s['y'] + '\t' + sentence + '\t###END###' + '\n'
            else:
                line = s['e1_id'] + '\t' + s['e2_id'] + '\t' + s['e1'] + '\t' + s['e2'] + '\t' + s['y'] + '\t' + sentence + '\t###END###' + '\n'
            ftrain.write(line)

    print "add tripples:", tpnew
    print 'Finally, considerting the repetitive sentences, we find %d false postive sentences.' %(rs_num)


def check_grad(model):
    for conv in model.convs1:
        print conv.weight.requires_grad
    print model.embed.weight.requires_grad
    print model.embed_pf1.weight.requires_grad
    print model.fc1.weight.requires_grad
