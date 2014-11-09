#!/usr/bin/env python
# coding: utf-8

from models import MatrixPreferenceDataModel as Model
from metrics import pearson_correlation as metrix
from similarities import UserSimilarity as Similarity
from recommenders import UserBasedRecommender as Recommender
from evaluations import MAE, RMSE

# 取整。这里以1.0为一个单位
def rounding(value):
    return int(value)

data = []
with open('../data-rs/80train.txt', 'r') as f :
    for line in f :
        datum = line.split("\t")
        data.append({ "user":datum[0], "movie":datum[1], "value":datum[2]})

# start running the data mining engine
model = Model(data)
similarity = Similarity(model, metrix)
recommender = Recommender(model, similarity)

# store the result in results.txt and compare them with test dataset
deviation = []
with open('../results.txt', 'w') as results, \
        open('../data-rs/test.txt', 'r') as test :
    for line in test :
        datum = line.split("\t")
        user, movie, value = recommender.recommend(datum[0], datum[1])
        value = rounding(value)
        results.write("%s\t%s\t%s\n" % (user, movie, value))
        deviation.append(abs(value - float(datum[2])))

# evaluate the output
MAE(deviation)
RMSE(deviation)
