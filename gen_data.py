# -*- coding: utf-8 -*-
import numpy as np
import cPickle as pickle
import sys, os
import pandas as pd
from collections import defaultdict

#if not os.path.exists('./data'):
#    os.mkdir('./data')

def confine_pf(num, max_pos):
    return max(0, min(num + max_pos, max_pos + max_pos + 1))

def load_data(ftrain, frela, fvec, args, data_path):
    """
    Transform all information from file ./origin_data/train.txt into a list-dict structure
    """
    max_sent, max_pos = args.max_sent, args.max_pos
    rela2id, word2id = {}, {}

    with open(frela, "r") as f:
        #rela_size = f.readline().strip()
        while True:
            line = f.readline()
            if line == '':
                break
            rela2id[line.strip().split()[0]] = int(line.strip().split()[1])
    pickle.dump(rela2id, open(data_path + 'rela2id.pkl', 'wb'))

    with open(fvec, "r") as f:
        W_size, w_dim = f.readline().strip().split()[:2]
        W_size, w_dim = int(W_size), int(w_dim)

        # word embedding matrix
        W = np.zeros(shape=(W_size+2, w_dim), dtype='float32')
        word2id['BLANK'] = 0  # the padding vector
        for i in range(W_size):
            temp = f.readline().strip().split()
            word2id[temp[0]] = i+1
            for j in range(w_dim):
                W[i+1][j] = (float)(temp[j+1])
        word2id['UNK'] = W_size+1
        W[W_size+1] = np.random.uniform(-0.25, 0.25, w_dim)  # UNK
    pickle.dump(word2id, open(data_path + 'word2id.pkl', 'wb'))
    np.savez(data_path + 'WordMatrix', W)

    instances = list()
    with open(ftrain, "r") as f:
        while True:
            line = f.readline()
            if line == '':
                break

            line = line.strip().split()
            e1_id = line[0]
            e2_id = line[1]
            e1 = line[2]
            e2 = line[3]
            rela = line[4]
            rela = 'NA' if rela not in rela2id else rela
            sent = line[5:-1]

            # Entity position
            e1_pos, e2_pos = 0, 0
            for num in range(len(sent)):
                if sent[num] == e1:
                    e1_pos = num
                if sent[num] == e2:
                    e2_pos = num

            instance  = {"y": rela,
                      "y_id": rela2id[rela],
                      "text": " ".join(sent),
                      "sent": sent,
                      "e1": e1,
                      "e1_id": e1_id,
                      "e2": e2,
                      "e2_id": e2_id,
                      "pf1": e1_pos,
                      "pf2": e2_pos,
                      "num_words": len(sent)}
            instances.append(instance)

            if len(instances)%100000 == 0:
                print "%d sentences have beeen processed ..." % (len(instances))
    pickle.dump(instances, open(data_path + 'instances.pkl', 'wb'))

    #tripple2sents = defaultdict(list)
    rela2tripple = defaultdict(set)
    for num in range(len(instances)):
        instance = instances[num]
        tripple = instance['e1_id'] + "\t" +instance['e2_id'] + "\t" + str(instance['y_id'])
        #tripple2sents[tripple].append(num)
        rela2tripple[str(instance['y_id'])].add(tripple)
    #pickle.dump(tripple2sents, open(data_path + 'tripple2sents.pkl', 'wb'))
    pickle.dump(rela2tripple, open(data_path + 'rela2tripple.pkl', 'wb'))
    #print "There are %d tripples in training set" % (len(tripple2sents))

    data_size = len(instances)
    sent2id = np.zeros((data_size, max_sent), dtype = np.int32) #  Maxtrix of word id in sentences
    pos2id_1 = np.zeros((data_size, max_sent), dtype = np.int32)
    pos2id_2 = np.zeros((data_size, max_sent), dtype = np.int32)
    label2id = np.zeros((data_size), dtype = np.int32)

    for s in range(data_size):
        sent = instances[s]['sent']
        pf1 = instances[s]['pf1']
        pf2 = instances[s]['pf2']
        label2id[s] = int(instances[s]['y_id'])
        for i in range(max_sent):
            pos2id_1[s][i] = confine_pf(i-pf1, max_pos)
            pos2id_2[s][i] = confine_pf(i-pf2, max_pos)
            if i < len(sent):
                if sent[i] not in word2id:
                    sent2id[s][i] = word2id['UNK']
                else:
                    sent2id[s][i] = word2id[sent[i]]

    inputs = np.concatenate((sent2id, pos2id_1, pos2id_2), 1)
    np.savez(data_path + "trans_id", sentM=sent2id, labelM=label2id, pf1M=pos2id_1, pf2M=pos2id_2)
    np.savez(data_path + "inputs", inputs=inputs)

    max_l = np.max(pd.DataFrame(instances)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(sent2id.shape[0])
    print "Word Embedding matrix size: " + str(W.shape)
    print "max sentence length: " + str(max_l)
    print "sentM:%s, labelM:%s, pf1M:%s, pf2M:%s" % (str(sent2id.shape), str(label2id.shape), str(pos2id_1.shape), str(pos2id_2.shape))
    print "dataset created!"

    RL_relations, rela2sents, sent2sents = process_data(instances, data_path, label2id)
    repetitive_tripples = find_repetitive_tripples(rela2id)

    return instances, inputs, label2id, W, RL_relations, rela2sents, sent2sents, repetitive_tripples

def process_data(instances, data_path, label2id):
    """
    Generate RL training dataset for each relation that 
    has more than 1500 sentences
    ep2sents 293162
    sent2sents 510386
    label2id shape (570088,)
    """
    ep2sents, sent2sents, rela2sents = dict(), dict(), dict()
    for num in range(len(instances)):
        sent_id = num
        ep = str(instances[sent_id]["e1_id"]) + '-' + str(instances[sent_id]["e2_id"]) + '-' + str(instances[sent_id]["y_id"])
        if ep2sents.has_key(ep):
            ep2sents[ep].append(sent_id)
        else:
            ep2sents[ep] = [sent_id]
    pickle.dump(ep2sents, open(data_path + "ep2sents.pkl", "wb"))
    print "There are %d entity pairs in training set" % (len(ep2sents))

    # For repetitive sentences of one entity pair, we just treat them as one sentence during reinforcement training
    for ep, sids in ep2sents.items():
        if len(sids) == 0:
            sys.exit('Error')
        elif len(sids) == 1:
            sent2sents[sids[0]] = sids
        else:
            stat_rpt = dict()
            for sid in sids:
                sent_text = instances[sid]["text"]
                if stat_rpt.has_key(sent_text):
                    stat_rpt[sent_text].append(sid)
                else:
                    stat_rpt[sent_text] = [sid]
            for rpt_sids in stat_rpt.values():
                sent2sents[rpt_sids[0]] = rpt_sids
    pickle.dump(sent2sents, open(data_path + "sent2sents.pkl", "wb"))
    print "There are %d unique sentences in the traning set" % len(sent2sents)

    tripple2sents = defaultdict(list)
    for num in range(len(instances)):
        instance = instances[num]
        tripple = instance['e1_id'] + "\t" +instance['e2_id'] + "\t" + str(instance['y_id'])
        tripple2sents[tripple].append(num)
    pickle.dump(tripple2sents, open(data_path + 'tripple2sents.pkl', 'wb'))
    print "There are %d tripples in training set" % (len(tripple2sents))

    #label2id = np.load(data_path + "trans_id.npz")['labelM']
    for sid in sent2sents.keys():
        rela_id = label2id[sid]
        if rela2sents.has_key(rela_id):
            rela2sents[rela_id].append(sid)
        else:
            rela2sents[rela_id] = [sid]

    RL_relations = list()
    for k, v in rela2sents.items():
        if len(v) > 1500 and k != 0:
            RL_relations.append(k)
            print k, len(v)
    pickle.dump(RL_relations, open(data_path + "RL_relations.pkl", "wb"))
    pickle.dump(rela2sents, open(data_path + "rela2sents.pkl", "wb"))
    print 'There are %d relations in the training data' % (len(rela2sents))
    print '%d relations will be processed by this system, they are %s' % (len(RL_relations), str(RL_relations))

    return RL_relations, rela2sents, sent2sents

def find_repetitive_tripples(rela2id):
    test = open('./origin_data/test.txt').readlines()
    test_dict = dict()
    for line in test:
        line = line.strip().split()
        e1_id, e2_id = line[0], line[1]
        rela = 'NA' if line[4] not in rela2id else line[4]
        key = e1_id + "\t" + e2_id + "\t" + str(rela2id[rela])
        test_dict[key] = 1

    train = open('./origin_data/train.txt').readlines()
    train_dict = dict()
    count = 0
    repetitive_tripples = dict()
    for line in train:
        line = line.strip().split()
        e1_id, e2_id = line[0], line[1]
        rela = 'NA' if line[4] not in rela2id else line[4]
        key = e1_id + "\t" + e2_id + "\t" + str(rela2id[rela])
        if test_dict.has_key(key):
            repetitive_tripples[key] = 1
            count += 1
    print "The size of repetitive_tripples: ", len(repetitive_tripples)
    pickle.dump(repetitive_tripples, open("./data/repetitive_tripples.pkl", "w"))

    return repetitive_tripples
