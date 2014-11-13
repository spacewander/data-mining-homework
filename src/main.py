#!/usr/bin/env python
# coding: utf-8

import models
import similarities
from metrics import manhattan_distances as metrix
import recommenders
from evaluations import MAE, RMSE

# 取整。这里以1.0为一个单位
def rounding(value):
    return round(value)

def recommenderFactory(data, Recommender):
    if Recommender.__base__ == recommenders.CFRecommender :
        if Recommender.__name__ == 'ItemCFRecommender' :
            Model = models.ItemPreferenceDataModel
            Similarity = similarities.ItemSimilarity
        elif Recommender.__name__ == 'UserCFRecommender' :
            Model = models.UserPreferenceDataModel
            Similarity = similarities.UserSimilarity

        model = Model(data)
        # num_best可以调，但是太少的话就得不到所需的数据了。
        # train dataset中共有用户943人，电影1650部
        similarity = Similarity(model, metrix)
        recommender = Recommender(model, similarity)
    else :
        if Recommender.__name__ == 'ItemBasedRecommender' :
            Model = models.ItemPreferenceDataModel
        elif Recommender.__name__ == 'UserBasedRecommender' :
            Model = models.UserPreferenceDataModel

        model = Model(data)
        recommender = Recommender(model)
    return recommender

def get_dataset():
    data = {}
    with open('../data-rs/80train.txt', 'r') as f :
        for line in f :
            datum = line.split("\t")
            if not data.has_key(datum[0]) :
                data[datum[0]] = {}
            data[datum[0]][datum[1]] = datum[2]
    return data

# 这里开始作为main函数
if __name__ == '__main__' :
    data = get_dataset()
    # start running the data mining engine
    #recommender = recommenderFactory(data, recommenders.UserCFRecommender)
    recommender = recommenderFactory(data, recommenders.ItemCFRecommender)
    #recommender = recommenderFactory(data, recommenders.MovieBasedRecomender)
    #recommender = recommenderFactory(data, recommenders.UserBasedRecommender)
    #recommender = recommenderFactory(data, recommenders.GoodOldManBasedRecommender)
    #recommender = recommenderFactory(data, recommenders.LuckBasedRecommender)

    # store the result in results.txt and compare them with test dataset
    deviation = []
    with open('../results.txt', 'w') as results, \
            open('../data-rs/test.txt', 'r') as test :
        i = 0
        for line in test :
            i += 1
            datum = line.split("\t")
            user, movie, value = recommender.recommend(datum[0], datum[1])
            value = rounding(value)
            results.write("%s\t%s\t%s\n" % (user, movie, value))
            deviation.append(abs(value - float(datum[2])))
            if i % 201 == 200 :
                print "已处理：第%d条\t%s\t%s\t%s\t%s" % (i, user, movie, value, datum[2])
                MAE(deviation)
                RMSE(deviation)
    # evaluate the output
    MAE(deviation)
    RMSE(deviation)

