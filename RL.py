# -*- coding: utf-8 -*-
from __future__ import division
import random
import numpy as np
from copy import deepcopy
from Networks import RC_CNN
from utils import *
from pretrain import pretrain_policy_CNN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor


class Denoiser(object):

    def __init__(self, args, embeddings, inputs, pos_data, neg_data):
        super(Denoiser, self).__init__()
        for k, v in vars(args).items(): setattr(self, k, v)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        #random.seed(args.seed)
        #np.random.seed(args.seed)
        torch.backends.cudnn.deterministic=True

        self.args = args
        self.embeddings = embeddings
        self.inputs = inputs
        self.pos_data = pos_data
        self.neg_data = neg_data

        self.pos_size = len(self.pos_data)
        if self.pos_size < 10000:
            self.neg_data = neg_data[:100000]
        else:
            self.neg_data = neg_data
        self.neg_size = min(10*self.pos_size, len(self.neg_data))
        #self.neg_size = max(10*self.pos_size, 30000)

        rv_initial = torch.zeros(1,100)
        self.rv_initial = Variable(rv_initial.type(FloatTensor))

        self.split_pos_data()
        self.split_neg_data()

        pretrain_data_size = self.pos_size + len(self.pretrain_neg)
        self.pretrain_batch_size = min(int(pretrain_data_size/100)+1, 1500)

        print '\n## PRETRAINING ##'
        self.Policy_model = pretrain_policy_CNN(self.pos_data, self.pretrain_neg, self.inputs, \
                                                self.embeddings, self.args, self.rv_initial, self.pretrain_batch_size)

        for conv in self.Policy_model.convs1:
            conv.weight.requires_grad = False

        Policy_parameters = filter(lambda p: p.requires_grad, self.Policy_model.parameters())
        self.Policy_optimizer = optim.RMSprop(Policy_parameters, lr=self.learning_rate)

        self.train_fix_remove, self.test_fix_remove = None, None
        self.alpha, self.F1_max, self.epoch_best = 2.0, 0.0, 0
        self.actions_best = list()
        self.Policy_best = deepcopy(self.Policy_model)
        self.rv_best = deepcopy(self.rv_initial)

    def split_pos_data(self):
        used_pos = self.pos_data if len(self.pos_data) < 10000 else random.sample(self.pos_data, 7800)
        self.train_pos, self.test_pos = split_data(used_pos)
        self.train_size = len(self.train_pos+self.test_pos)
        print "For RL training, train positive: %d, test positive: %d" % (len(self.train_pos), len(self.test_pos))

    def split_neg_data(self):
        self.used_neg = random.sample(self.neg_data, self.neg_size)
        train_neg_, test_neg_ = split_data(self.used_neg)
        self.train_neg = random.sample(train_neg_, 2*len(self.train_pos))
        self.test_neg = random.sample(test_neg_, 2*len(self.test_pos))
        print "For RL training, train negative: %d, test negative: %d" % (len(self.train_neg), len(self.test_neg))
        self.pretrain_neg = list(set(self.used_neg) - set(self.train_neg) - set(self.test_neg))
        print "For RL training, pretraining negative: %d" % (len(self.pretrain_neg))

    def select_sentences(self, model, x, rv):
        trainloader = generate_trainloader(self.inputs, x, [1]*len(x), 1000, shuf=False)
        actions = list()
        for i, (x_, _) in enumerate(trainloader, 0):
            _, _, _, actions_ = self.select_action(model, x_, rv)
            actions += actions_
        x, actions = np.array(x), np.array(actions)
        #print x.shape, actions.shape
        assert x.shape == actions.shape
        sents_retain = list(x[np.where(actions == 1)[0]])
        sents_remove = list(x[np.where(actions == 0)[0]])
        return sents_remove, sents_retain

    def select_action(self, model, x, rv):
        model.eval()
        rv_matrix = rv.repeat(x.size(0), 1)
        logits, sent_vecs = calculate_logits(model, x, self.max_sent, rv_matrix=rv_matrix)
        probs = F.softmax(logits, dim=1).view(x.size(0), -1)
        probs = probs.data.type(torch.FloatTensor).numpy()
        remove_probs = probs[:,0]

        # Actions sampled by probs
        actions_prob = list()
        for prob in probs:
            if abs(prob[0]-prob[1])<0.1:
                actions_prob.append(int(np.random.choice(np.arange(2), p=prob)))
            else:
                actions_prob.append(int(np.argmax(prob)))

        # Actual actions
        actions = list(np.argmax(probs, axis=1))

        remove_idx = np.where(np.array(actions_prob) == 0)[0]
        if len(remove_idx) == 0:
            return actions_prob, list(remove_probs), torch.zeros(1,100).type(FloatTensor), actions
        else:
            return actions_prob, list(remove_probs), torch.mean(sent_vecs.data[remove_idx], 0, True), actions

    def retrain_relation_classifier(self, actions):

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        RC_model = RC_CNN(self.embeddings, self.args, emb_change=False, conv_change=False)
        RC_model = RC_model.cuda() if use_cuda else RC_model

        para_dict = torch.load('./models/Policy_pretrain.pkl')

        layers = ['embed.weight', 'embed_pf1.weight', 'embed_pf2.weight', 'convs1.0.weight', 'convs1.0.bias']
        pretrained_dict = {k:v for k,v in para_dict.items() if k in layers}
        model_dict = RC_model.state_dict()
        model_dict.update(pretrained_dict)
        RC_model.load_state_dict(model_dict)

        #check_grad(RC_model)

        actions = list(actions)
        label_train, label_test = actions[:len(self.train_pos)], actions[len(self.train_pos):]
        data_size = len(self.train_pos) + len(self.train_neg)
        x_ = self.train_pos + self.train_neg
        y_ = label_train + [0]*len(self.train_neg)
        trainloader = generate_trainloader(self.inputs, x_, y_, int(data_size/20)+1, shuf=True)
        RC_parameters = filter(lambda p: p.requires_grad, RC_model.parameters())
        RC_optimizer = optim.Adam(RC_parameters)

        F1_sum = 0.0
        weights = [1.0, 1.0]
        for epoch in range(6):
            RC_model.train()
            for i, (x, y) in enumerate(trainloader, 0):
                logits = calculate_logits(RC_model, x, self.max_sent)
                loss = calculate_loss(logits, y, weights)
                RC_optimizer.zero_grad()
                loss.backward()
                RC_optimizer.step()

            if epoch > 0:
                x_ = self.test_pos+self.test_neg
                y_ = label_test+[0]*len(self.test_neg)
                F1 = calculate_F1(RC_model, x_, y_, self.inputs, self.max_sent)
                F1_sum += F1

        return float(F1_sum/5)

    def fix_number(self, actions, remove_probs):
        boundry = len(self.train_pos)
        actions_train, actions_test = deepcopy(actions[:boundry]), deepcopy(actions[boundry:])
        prob_train, prob_test = deepcopy(remove_probs[:boundry]), deepcopy(remove_probs[boundry:])
        if len(np.where(actions_train == 0)[0]) > self.train_fix_remove:
            boundry_prob = np.sort(prob_train)[-self.train_fix_remove]
            actions_train[np.where(prob_train < boundry_prob)[0]] = 1

        if len(np.where(actions_test == 0)[0]) > self.test_fix_remove:
            boundry_prob = np.sort(prob_test)[-self.test_fix_remove]
            actions_test[np.where(prob_test < boundry_prob)[0]] = 1
        return np.concatenate((actions_train, actions_test)), len(np.where(actions_train == 0)[0]), len(np.where(actions_test == 0)[0])

    def reinforcement_learning(self):

        PMD_batch = int(self.train_size/50) + 1
        x_ = self.train_pos+self.test_pos
        y_ = [1]*len(x_)
        trainloader = generate_trainloader(self.inputs, x_, y_, PMD_batch, shuf=False)

        actions_prev = np.array(y_)
        F1_prev = self.retrain_relation_classifier(actions_prev)

        rv = deepcopy(self.rv_initial)

        for epoch in range(self.max_epoch):

            actions_prob, actions = list(), list()
            remove_probs = list()
            for i, (x, _) in enumerate(trainloader, 0):
                if i == 0:
                    rv_matrix = rv.repeat(x.size(0), 1)
                else:
                    rv_matrix = torch.cat((rv_matrix, rv.repeat(x.size(0), 1)), 0)
                actions_prob_, remove_probs_, cur_rv, actions_ = self.select_action(self.Policy_model, x, rv)
                actions_prob += actions_prob_
                actions += actions_
                remove_probs += remove_probs_
                cur_rv = 0.01 * Variable(cur_rv)
                rv = (rv * i + cur_rv)/float(i+1)
            actions_prob, actions = np.array(actions_prob), np.array(actions)
            remove_probs = np.array(remove_probs)

            if epoch == 0:
                self.train_fix_remove = len(self.train_pos) - np.sum(actions_prob[:len(self.train_pos)])
                self.test_fix_remove = len(self.test_pos) - np.sum(actions_prob[len(self.train_pos):])
                train_remove_num, test_remove_num = self.train_fix_remove, self.test_fix_remove
            else:
                actions_prob, train_remove_num, test_remove_num = self.fix_number(actions_prob, remove_probs)


            F1 = self.retrain_relation_classifier(actions_prob)
            F1_criterion = self.retrain_relation_classifier(actions)
            remove_actual_num, retain_actual_num = len(np.where(actions == 0)[0]), len(np.where(actions == 1)[0])
            # sentece index
            remove_prev, retain_prev = np.where(actions_prev == 0)[0], np.where(actions_prev == 1)[0]
            remove_curr, retain_curr = np.where(actions_prob == 0)[0], np.where(actions_prob == 1)[0]

            # remove part
            same_part = list(set(remove_prev)&set(remove_curr))
            diff_part_prev = list(set(remove_prev)-set(same_part))
            diff_part_curr = list(set(remove_curr)-set(same_part))

            if epoch != 0:
                reward = (F1 - F1_prev) * self.reward_scale
            else:
                reward = -0.01

            all_sents = np.array(self.train_pos + self.test_pos)

            loss_1, loss_2 = Variable(torch.cuda.FloatTensor([0])), Variable(torch.cuda.FloatTensor([0]))

            self.Policy_model.train()

            if len(diff_part_prev) != 0:
                x_prev = torch.LongTensor(self.inputs[all_sents[diff_part_prev]])
                y_prev = [0]*len(diff_part_prev)
                y_prev = LongTensor(y_prev)
                logits_prev, _ = calculate_logits(self.Policy_model, x_prev, self.max_sent, rv_matrix=rv_matrix[diff_part_prev])
                loss_1 = calculate_loss(logits_prev, y_prev)
                loss_1 = (-reward) * loss_1
            if len(diff_part_curr) != 0:
                x_curr = torch.LongTensor(self.inputs[all_sents[diff_part_curr]])
                y_curr = [0]*len(diff_part_curr)
                y_curr = LongTensor(y_curr)
                logits_curr, _ = calculate_logits(self.Policy_model, x_curr, self.max_sent, rv_matrix=rv_matrix[diff_part_curr])
                loss_2 = calculate_loss(logits_curr, y_curr)
                loss_2 = reward * loss_2

            if reward > 0:
                loss = self.alpha * loss_1 + loss_2
            else:
                loss = loss_1 + self.alpha * loss_2

            print '[Epoch %d] Cur_F1: %.4f, Pre_F1: %.4f, reward: %.4f, remove_part:[%d, %d], diff_part: [%d, %d], F1_criterion: %.4f, remove_num: %d' \
                   %(epoch, F1, F1_prev, reward, train_remove_num, test_remove_num, len(diff_part_prev), len(diff_part_curr), F1_criterion, remove_actual_num)

            if loss.data[0] != 0:
                self.Policy_optimizer.zero_grad()
                loss.backward()
                self.Policy_optimizer.step()
            else:
                'NO LOSS!!!!!!!!!!!!!!!!!!!'

            actions_prev = actions_prob
            F1_prev = F1

            if F1_criterion > self.F1_max:
                self.F1_max = F1_criterion
                self.epoch_best = epoch
                self.actions_best = actions
                self.Policy_best = deepcopy(self.Policy_model)
                self.rv_best = rv
                print 'MAX: epoch: %d, F1_criterion: %.4f' %(epoch, F1_criterion)

            if epoch - self.epoch_best > 30:
                break
                print 'Early Stop!!'

        if self.pos_size > 10000:
            sents_remove_best, sents_retain_best = self.select_sentences(self.Policy_best, self.pos_data, self.rv_best)
            #select_sentences(self, model, x, rv)
        else:
            sents_retain_best = list(all_sents[np.where(self.actions_best == 1)[0]])
            sents_remove_best = list(all_sents[np.where(self.actions_best == 0)[0]])
        print "***ACTUAL FINISH: Sent_remove: %d, Sent_retain: %d" %(len(sents_remove_best), len(sents_retain_best))

        return sents_remove_best, sents_retain_best
