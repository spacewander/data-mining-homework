# coding: utf-8

import numpy as np
from utils import UserNotFoundError, ItemNotFoundError

class UserPreferenceDataModel(object):
    """
    build data model according to user's preferences
    """
    def __init__(self, data):
        self.data = data
        self.build_model()

    def __getitem__(self, user_id):
        return self.preferences_from_user(user_id)

    def __iter__(self):
        for index, user in enumerate(self.user_ids):
            yield user, self[user]

    def __len__(self):
        return self.index.shape

    def build_model(self):
        self.user_ids = np.asanyarray(self.data.keys())
        self.user_ids.sort()

        self.item_ids = []
        for items in self.data.itervalues():
            self.item_ids.extend(items.keys())

        self.item_ids = np.unique(np.array(self.item_ids))
        self.item_ids.sort()

        self.index = np.empty(shape=(self.user_ids.size, self.item_ids.size))
        for user_no, user_id in enumerate(self.user_ids):
            for item_no, item_id in enumerate(self.item_ids):
                # 如果该用户没有看过某电影，设置为NaN而不是0
                r = self.data[user_id].get(item_id, np.NaN)
                self.index[user_no, item_no] = r

    def preference_values_from_user(self, user_id):
        """
        return: numpy.ndarray
        """
        found_user_id = np.where(self.user_ids == user_id)
        if not found_user_id[0].size:
            # user_id not found
            raise UserNotFoundError('你所查找的用户不存在')

        preferences = self.index[found_user_id]

        return preferences

    def preferences_from_user(self, user_id):
        preferences = self.preference_values_from_user(user_id)

        data = zip(self.item_ids, preferences.flatten())

        return [(item_id, preference)  for item_id, preference in data \
                         if not np.isnan(preference)]

    def preference_value(self, user_id, item_id):
        '''
        return a float preference
        '''
        found_item_id = np.where(self.item_ids == item_id)
        found_user_id = np.where(self.user_ids == user_id)

        if not found_user_id[0].size:
            raise UserNotFoundError('你所查找的用户不存在')

        if not found_item_id[0].size:
            raise ItemNotFoundError('你所查找的项目不存在')

        return self.index[found_user_id, found_item_id].flatten()[0]

    def items_count(self):
        return self.item_ids.size

    def items_from_user(self, user_id):
        preferences = self.preferences_from_user(user_id)
        return [key for key, value in preferences]

    def users_count(self):
        return self.user_ids.size

    def max_preference_value(self):
        return 5.0

    def min_preference_value(self):
        return 1.0

