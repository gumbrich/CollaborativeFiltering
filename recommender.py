# Collaborative Filtering algorithm applied to MovieLens 100k Dataset
# date: June 22, 2021

import numpy as np
import pandas as pd
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=header)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print("Number of users = " + str(n_users) + ", and number of movies = " + str(n_items))

train_data, test_data = train_test_split(df, test_size=0.25)

train_data_matrix, test_data_matrix = np.zeros((n_users, n_items)), np.zeros((n_users, n_items))

for line in train_data.itertuples():
	train_data_matrix[line[1]-1, line[2]-1] = line[3]
	
for line in test_data.itertuples():
	test_data_matrix[line[1]-1, line[2]-1] = line[3]
	
# simil = cos(theta) = scalar_prod(a,b)/(norm(a)*norm(b))
user_simil = pairwise_distances(train_data_matrix, metric='cosine')
item_simil = pairwise_distances(train_data_matrix.T, metric='cosine')

# user-based collaborative filtering: P_{u,i} = \sum_v R_{v,i} * simil_{u,v} / \sum_v simil_{u,v}
def predict(ratings, similarity, type='user'):
	if type=='user':
		mean_user_rating = ratings.mean(axis=1)
		# make the format of mean_user_rating compatible with the format of ratings: numpy.newaxis
		ratings_diff = (ratings - mean_user_rating[:,np.newaxis])
		prediction = mean_user_rating[:,np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
	elif type=='item':
		prediction = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
	return prediction

item_prediction = predict(train_data_matrix, item_simil, type='item')
user_prediction = predict(train_data_matrix, user_simil, type='user')

def rms_error(prediction, ground_truth):
	pred = prediction[ground_truth.nonzero()].flatten()
	base = ground_truth[ground_truth.nonzero()].flatten()
	return math.sqrt(mean_squared_error(prediction, ground_truth))

print("Item-based Collaborative Filtering RMS Error " + str(rms_error(item_prediction, test_data_matrix)))
print("User-based Collaborative Filtering RMS Error " + str(rms_error(user_prediction, test_data_matrix)))
