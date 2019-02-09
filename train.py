# -*- coding: utf-8 -*-
import sys, os
import random
import numpy as np
from args import load_hyperparameters
from utils import *
from gen_data import load_data
from RL import Denoiser
import cPickle as pickle
import torch


# "./models" saves the intermediate model files
if not os.path.exists('./models'):
    os.mkdir('./models')

# "./data" saves the processed data files
if not os.path.exists('./data'):
    os.mkdir('./data')

input_path = "./origin_data/"
data_path = "./data/"

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor


if __name__ == "__main__":
    args = load_hyperparameters()
    # if gpu is to be used
    if use_cuda:
        torch.cuda.set_device(args.device)
        print "GPU is available!"
    else:
        print "GPU is not available!"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic=True

    # Loading data
    print 'Loading data ......'
    fvec = input_path + 'vec.txt'
    ftrain = input_path + 'train.txt'
    frela = input_path + 'relation2id.txt'

    instances, inputs, labels, embeddings, RL_relations, rela2sents, sent2sents, repetitive_tripples = load_data(ftrain, frela, fvec, args, data_path)
    embeddings = torch.from_numpy(embeddings.astype(np.float64))
    RL_train_data = filter_negative(embeddings, inputs, rela2sents, args)

    """
    embeddings = np.load(data_path + 'WordMatrix.npz')['arr_0']
    embeddings = torch.from_numpy(embeddings.astype(np.float64))
    inputs = np.load(data_path + 'inputs.npz')['inputs']
    rela2sents = pickle.load(open(data_path + 'rela2sents.pkl'))
    RL_relations = pickle.load(open('./data/RL_relations.pkl'))
    RL_train_data = pickle.load(open(data_path + 'RL_train_data.pkl'))
    """

    false_positive = list()
    RL_remove = list()
    neg_data = RL_train_data[0]
    for relaid in RL_relations:

        print '\n############## RELATION %s is being denoised ##############' % (relaid)

        pos_data = RL_train_data[relaid]

        denoiser = Denoiser(args, embeddings, inputs, pos_data, neg_data)

        print '\n## REINFORCEMENT LEARNING ##'
        sents_remove_best, sents_retain_best = denoiser.reinforcement_learning()

        false_positive += sents_remove_best
        RL_remove += sents_remove_best

    print '\n\n############## Statistical Result ##############'
    print 'In total, we find %d false postive unique sentences, in which RL remove %d sentences.' % (len(false_positive), len(RL_remove))
    pickle.dump(false_positive, open(data_path + 'false_positive.pkl', 'wb'))

    # Change origin the labels of Distant-Supervised dataset
    instances = pickle.load(open(data_path + "instances.pkl"))
    false_positive = pickle.load(open(data_path + 'false_positive.pkl'))
    repetitive_tripples = pickle.load(open(data_path + 'repetitive_tripples.pkl'))
    sent2sents = pickle.load(open(data_path + 'sent2sents.pkl'))
    tripple2sents = pickle.load(open(data_path + 'tripple2sents.pkl'))
    rela2tripple = pickle.load(open(data_path + 'rela2tripple.pkl'))
    generate_new_dataset(sent2sents, instances, false_positive, repetitive_tripples, tripple2sents, rela2tripple, args, fname='train.txt', train_version=args.train_version)

