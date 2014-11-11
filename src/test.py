#!/usr/bin/env python
# coding: utf-8

# this script is used to test whether everything is ok
from main import get_dataset
from similarities import UserSimilarity
from models import UserPreferenceDataModel
from metrics import pearson_correlation
from recommenders import UserCFRecommender as Recommender

data = get_dataset()
model = UserPreferenceDataModel(data)
sim = UserSimilarity(model, pearson_correlation)

recommender = Recommender(model, sim)
print recommender.recommend('244', '51')
