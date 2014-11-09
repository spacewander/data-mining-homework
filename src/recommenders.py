from random import randint

class Recommender(object):
    def __init__(self, model, similarity):
        self.model = model
        self.similarity = similarity

class UserBasedRecommender(Recommender):
    def recommend(self, user, movie):
        return (user, movie, 3.0)

class GoodOldManBasedRecommender(Recommender):
    def recommend(self, user, movie):
        return (user, movie, 3.0)

class LuckBasedRecommender(Recommender):
    def recommend(self, user, movie):
        return (user, movie, randint(1, 5))
