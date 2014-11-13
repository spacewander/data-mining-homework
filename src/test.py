#!/usr/bin/env python
# coding: utf-8

# this script is used to test whether everything is ok
import cPickle
import os.path
from main import get_dataset
from similarities import ItemSimilarity
from models import ItemPreferenceDataModel
from metrics import manhattan_distances
#from recommenders import ItemCFRecommender as Recommender

data = get_dataset()
model = ItemPreferenceDataModel(data)
sim = ItemSimilarity(model, manhattan_distances)
dump_file = 'cache_' + sim.__class__.__name__ + '_' + \
                sim.metrix.__name__

def save_result(sim, filename):
    with open(filename, 'w') as f :
        cPickle.dump(sim.similarities, f)


def load_similarity(sim, filename):
    if os.path.exists(filename) :
        with open(filename, 'r') as f :
            sim.similarities = cPickle.load(f)
    else :
        sim.similarities = {}

load_similarity(sim, dump_file)
save_result(sim, dump_file)
