import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances



#Return user_profile for given user_id
def get_user_profile_by_user_id(users_profiles,user_id):
    #Getting the user vector profile#
    if users_profiles[users_profiles['user_id']==user_id].drop(columns=['user_id']).shape[0]==0:
        return users_profiles.mean().to_frame().transpose().drop(columns=['user_id'])
    else:
        return users_profiles[users_profiles['user_id']==user_id].drop(columns=['user_id'])

#Return item_name for given item_id
def get_items_names(ids_items_dict, items_list):
    top_items_names = [ids_items_dict[str(it_id)] for it_id in items_list]
    return pd.DataFrame(top_items_names,columns=['itemName'])
    #return items_ids[items_ids['item_id'].isin(items_list)]


#Get top_n recom
def get_top_n(user_id,trainset,cf_algo, n=10):
    # Get a list of all items and all users in the dataset
    items = trainset.all_items()

    # Convert raw ids to inner ids
    raw_user_id = user_id  # replace with the user ID you are interested in
    inner_user_id = trainset.to_inner_uid(raw_user_id)

    # Find items that the user has not rated yet
    test_items = [trainset.to_raw_iid(item) for item in items if not trainset.ur[inner_user_id].__contains__(item)]

    # Predict ratings for all the items not rated by the user
    predictions = [cf_algo.predict(raw_user_id, item) for item in test_items]

    # Get the top N recommendations
    top_n_items = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    return top_n_items


# Get top n recommendation according to content based
def get_top_k_items_for_specific_user(top_k,user_profile,user_items_ids,items_profiles,dis_metric):
    #Filter only unseen items 
    unseen_items = items_profiles[~items_profiles['item_id'].isin(user_items_ids)]
    
    if dis_metric=='CB_Cosine':
        items_scores = cosine_similarity(user_profile,unseen_items.iloc[:, :-1].values)
        #Looking for the higer cosin simillarity#
        top_items_scores = np.sort(items_scores)[0][::-1][:top_k]
        ratings = [cosine_similarity_to_rating(score) for score in top_items_scores]
        top_items_indices = np.argsort(items_scores)[0][::-1][:top_k]
    elif dis_metric=='CB_Euclidean':
        items_scores = euclidean_distances(user_profile,unseen_items.iloc[:, :-1].values)
        #Looking for the lower euclidean#
        top_items_scores = np.sort(items_scores)[0][:top_k]
        ratings = [euclidean_to_rating(score) for score in top_items_scores]
        top_items_indices = np.argsort(items_scores)[0][:top_k]
    
    top_items_ids = unseen_items.iloc[top_items_indices]['item_id'].values
    return top_items_ids,top_items_scores,ratings

# Convert the cosine similarity score to rating
def cosine_similarity_to_rating(cosine_similarity):   
    normalized_score = (cosine_similarity + 1) / 2
    rating = 1 + (normalized_score * 4)
    return rating

# Convert euclidean score to rating
def euclidean_to_rating(euclidean_dist):
    normalized_score = 1/(euclidean_dist + 1) 
    rating = 1 + (normalized_score * 4)
    return rating