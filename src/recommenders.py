# coding: utf-8

from random import randint
import os.path
import cPickle
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


class MovieBasedRecomender(Recommender):
    def __init__(self, model):
        super(MovieBasedRecomender, self).__init__(model)
        self.mean_from_item = {}

    def recommend(self, user_id, item_id):
        try :
            return self._recommend(user_id, item_id)
        except (UserNotFoundError, ItemNotFoundError) :
            print 'lost!'
            return (user_id, item_id, 4.0)

    def _recommend(self, user_id, item_id):
        if not self.mean_from_item.has_key(item_id) :
            preferences = self.model.preference_values_from_item(item_id)
            # 去掉NaN
            preferences = preferences[~np.isnan(preferences)]
            self.mean_from_item[item_id] = np.mean(preferences)

        return (user_id, item_id, self.mean_from_item[item_id])


class UserBasedRecommender(Recommender):
    def __init__(self, model):
        super(UserBasedRecommender, self).__init__(model)
        self.mean_from_user = {}

    def recommend(self, user_id, item_id):
        try :
            return self._recommend(user_id, item_id)
        except (UserNotFoundError, ItemNotFoundError) :
            print 'lost!'
            return (user_id, item_id, 4.0)

    def _recommend(self, user_id, item_id):
        if not self.mean_from_user.has_key(user_id) :
            preferences = self.model.preference_values_from_user(user_id)
            # 去掉NaN
            preferences = preferences[~np.isnan(preferences)]
            self.mean_from_user[user_id] = np.mean(preferences)

        return (user_id, item_id, self.mean_from_user[user_id])


class CFRecommender(Recommender):
    def __init__(self, model, similarity):
        self.model = model
        self.similarity = similarity
        self.dump_file = 'cache_' + self.similarity.__class__.__name__ + '_' + \
                self.similarity.metrix.__name__

    def save_result(self, filename):
        with open(filename, 'w') as f :
            cPickle.dump(self.similarity.similarities, f)


    def load_similarity(self, filename):
        if os.path.exists(filename) :
            with open(filename, 'r') as f :
                self.similarity.similarities = cPickle.load(f)
        else :
            self.similarity.similarities = {}


class UserCFRecommender(CFRecommender):
    """
    基于用户的协同过滤推荐器
    """
    def __init__(self, model, similarity):
        super(UserCFRecommender, self).__init__(model, similarity)
        self.load_similarity(self.dump_file)
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
        except Exception :
            print 'lost!'
            return (user_id, item_id, 4.0)

class ItemCFRecommender(CFRecommender):
    """
    基于项目的协同过滤推荐器
    """
    def __init__(self, model, similarity):
        super(ItemCFRecommender, self).__init__(model, similarity)
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

        # 获取相似项目群
        nearest_neighbors = self.nearest_neighbors.setdefault(item_id, \
                self.build_neighborhood(item_id, self.model, self.similarity.metrix))

        # 获取相似项目群内每位项目的相似度，已经过滤掉NaN了
        similarities = self.neighbors_similarity.setdefault(user_id, \
                np.array([similarity for neighbor, similarity in nearest_neighbors]))

        # 获取这些项目的打分
        preferences = np.array([self.model.preference_value(user_id, neighbor)
                 for neighbor, similarity in nearest_neighbors])

        # 去掉NaN
        preferences = preferences[~np.isnan(preferences)]
        similarities = similarities[~np.isnan(preferences)]

        # 根据各自的权重汇总，得出总打分
        prefs_sim = sum([prefs * sim for prefs, sim in zip(preferences, similarities)])
        #prefs_sim = np.sum(preferences[~np.isnan(similarities)] *
                             #similarities[~np.isnan(similarities)])
        total_similarity = sum(similarities)

        # 相似度无法判断
        if total_similarity == 0.0 or \
           not similarities[~np.isnan(similarities)].size:
            print 'lost!'
            return (user_id, item_id, 4.0)

        estimated = prefs_sim / total_similarity

        # 分数范围在1到5之间
        if estimated < 1 :
            estimated = 1
        elif estimated > 5 :
            estimated = 5

        return (user_id, item_id, estimated)

    def build_neighborhood(self, item_id, model, metrix):
        """
        return:
        [(neighbor1, similarity1), (neighbor2, similarity2), ...]
        The length of result is defined as num_best in Similarity
        """
        # 只能是正相关
        minimal_similarity = 0.2

        neighborhood = [(neighbor, score) for neighbor, score in self.similarity[item_id] \
                           if not np.isnan(score) and score >= minimal_similarity \
                                and item_id != neighbor]
        return neighborhood

    def recommend(self, user_id, item_id) :
        try :
            return self._recommend(user_id, item_id)
        except Exception:
            print 'not found!'
            return (user_id, item_id, 4.0)


