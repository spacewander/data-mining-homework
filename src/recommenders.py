# coding: utf-8

from random import randint
import numpy as np
from utils import UserNotFoundError, ItemNotFoundError

class Recommender(object):
    def __init__(self, model):
        self.model = model

    def recommend(self, user, movie):
        raise NotImplementedError("recommend method not defined")


class GoodOldManBasedRecommender(Recommender):
    def recommend(self, user, movie):
        return (user, movie, 4.0)

class LuckBasedRecommender(Recommender):
    def recommend(self, user, movie):
        return (user, movie, randint(1, 5))


class CFRecommender(Recommender):
    def __init__(self, model, similarity):
        self.model = model
        self.similarity = similarity


class UserCFRecommender(CFRecommender):
    """
    基于用户的协同过滤推荐器
    """
    def __init__(self, model, similarity):
        super(CFRecommender, self).__init__(model)
        self.similarity = similarity
        self.nearest_neighbors = {}
        self.neighbors_similarity = {}

    def _recommend(self, user_id, item_id):
        '''
        根据user_id和item_id来获取可能的分数
        return:
        user_id, item_id 和 一个打分
        1. 如果之前打过分，返回打分
        2. 如果相似度无法区分，返回4.0
        3. 返回根据相似度所评的分
        '''
        # 如果用户曾经给该项目打过分
        preference = self.model.preference_value(user_id, item_id)
        if not np.isnan(preference):
            return (user_id, item_id, preference)

        # 获取相似用户群
        nearest_neighbors = self.nearest_neighbors.setdefault(user_id, \
                self.build_neighborhood(user_id, self.model, self.similarity.metrix))

        preference = 0.0
        total_similarity = 0.0

        # 获取相似用户群内每位用户的相似度，已经过滤掉NaN了
        similarities = self.neighbors_similarity.setdefault(user_id, \
                np.array([similarity for neighbor, similarity in nearest_neighbors]))

        # 获取这些用户的打分
        preferences = np.array([self.model.preference_value(neighbor, item_id)
                 for neighbor, similarity in nearest_neighbors])

        # 去掉NaN
        preferences = preferences[~np.isnan(preferences)]
        similarities = similarities[~np.isnan(preferences)]

        # 根据各自的权重汇总，得出总打分
        prefs_sim = np.sum(preferences[~np.isnan(similarities)] *
                             similarities[~np.isnan(similarities)])
        total_similarity = np.sum(similarities)

        # 相似度无法判断
        if total_similarity == 0.0 or \
           not similarities[~np.isnan(similarities)].size:
            print 'lost!'
            return (user_id, item_id, 4.0)

        estimated = prefs_sim / total_similarity

        # 分数范围在1到5之间
        if estimated <= self.model.min_preference_value() :
            estimated = self.model.min_preference_value()
        elif estimated >= self.model.max_preference_value() :
            estimated = self.model.max_preference_value()

        return (user_id, item_id, estimated)

    def build_neighborhood(self, user_id, model, metrix):
        """
        return:
        [(neighbor1, similarity1), (neighbor2, similarity2), ...]
        The length of result is defined as num_best in Similarity
        """
        # 只能是正相关
        minimal_similarity = 0.0

        neighborhood = [(neighbor, score) for neighbor, score in self.similarity[user_id] \
                           if not np.isnan(score) and score >= minimal_similarity \
                                and user_id != neighbor]
        return neighborhood

    def recommend(self, user_id, item_id) :
        try :
            return self._recommend(user_id, item_id)
        except (UserNotFoundError, ItemNotFoundError) :
            print 'lost!'
            return (user_id, item_id, 4.0)


