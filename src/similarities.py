# coding: utf-8

import numpy as np

class Similarity(object):
    def __init__(self, model, metrix, num_best = None):
        self.model = model
        self.metrix = metrix
        self.num_best = num_best
        self.similarities = {}

    def __getitem__(self, source_id):
        """
        get relative similarities, the number of similarities is according to num_best.
        work as sim['196']
        Warning: this method will cause a lot of computing resource, so don't forget to cache it!
        """
        if not self.similarities.has_key(source_id) :
            sims = self.get_similarities(source_id)

            # 将结果从nparray转换成一个值
            if sims:
                item_ids, preferences = zip(*sims)
                preferences = np.array(preferences).flatten()
                item_ids = np.array(item_ids).flatten()
                sorted_prefs = np.argsort(-preferences)
                tops = zip(item_ids[sorted_prefs], preferences[sorted_prefs])

            # 以(id, preferences)形式输出前num_best个结果
            if self.num_best is not None :
                self.similarities[source_id] = tops[:self.num_best]
            else :
                self.similarities[source_id] = tops
        return self.similarities[source_id]


def find_common_elements(reference_preferences, target_preferences):
    '''
    Returns the preferences from both vectors
    '''
    ref = dict(reference_preferences)
    target = dict(target_preferences)

    inter = np.intersect1d(ref.keys(), target.keys())

    # 不要忘记过滤掉NaN
    common_preferences = zip(*[(ref[item], target[item]) for item in inter \
            if not np.isnan(ref[item]) and not np.isnan(target[item])])
    if common_preferences:
        return np.asarray([common_preferences[0]]), np.asarray([common_preferences[1]])
    else:
        return np.asarray([[]]), np.asarray([[]])

class UserSimilarity(Similarity):
    """
    get similarities between each two users
    """
    def __init__(self, model, metrix, num_best = None):
        super(UserSimilarity, self).__init__(model, metrix, num_best)

    def get_similarity(self, reference_id, target_id):
        """
        返回结果可能是个多维数组
        """
        reference_preferences = self.model.preferences_from_user(reference_id)
        target_preferences = self.model.preferences_from_user(target_id)

        reference_preferences, target_preferences = \
            find_common_elements(reference_preferences, target_preferences)

        if reference_preferences.ndim == 1 and target_preferences.ndim == 1:
            reference_preferences = np.asarray([reference_preferences])
            target_preferences = np.asarray([target_preferences])

        # evaluate the similarity between the two users vectors.
        if not reference_preferences.shape[1] == 0 \
                and not target_preferences.shape[1] == 0 :
            return self.metrix(reference_preferences, target_preferences)
        else:
            # 没有共同点
            return np.array([[np.nan]])

    def get_similarities(self, reference_id):
        similarities = [(other_id, self.get_similarity(reference_id, other_id))  \
                for other_id, _ in self.model]
        return similarities

    def __iter__(self):
        """
        用于上面的get_similarity函数中，实现迭代器所需
        """
        for reference_id, preferences in self.model:
            yield reference_id, self[reference_id]

class ItemSimilarity(Similarity):
    """
    get similarities between each two items
    """
    def __init__(self, model, metrix, num_best = None):
        super(ItemSimilarity, self).__init__(model, metrix, num_best)

    def get_similarity(self, reference_id, target_id):
        """
        返回结果可能是个多维数组
        """
        reference_preferences = self.model.preferences_from_item(reference_id)
        target_preferences = self.model.preferences_from_item(target_id)

        reference_preferences, target_preferences = \
            find_common_elements(reference_preferences, target_preferences)

        if reference_preferences.ndim == 1 and target_preferences.ndim == 1:
            reference_preferences = np.asarray([reference_preferences])
            target_preferences = np.asarray([target_preferences])

        # evaluate the similarity between the two users vectors.
        if not reference_preferences.shape[1] == 0 \
                and not target_preferences.shape[1] == 0 :
            return self.metrix(reference_preferences, target_preferences)
        else:
            # 没有共同点
            return np.array([[np.nan]])

    def get_similarities(self, reference_id):
        similarities = [(other_id, self.get_similarity(reference_id, other_id))  \
                for other_id, _ in self.model]
        return similarities

    def __iter__(self):
        """
        用于上面的get_similarity函数中，实现迭代器所需
        """
        for reference_id, preferences in self.model:
            yield reference_id, self[reference_id]


