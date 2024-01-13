import pandas as pd
import json
import pickle
import src.utils as ut


"""
General data - usernames and items ids
"""
with open('./outputs/IDs/ids_items_dict.json', 'r') as file:
    ids_items_dict = json.load(file)
with open('./outputs/IDs/usernames_ids_dict.json', 'r') as file:
    usernames_ids_dict = json.load(file)

"""
Content Based
"""
#Load user profiles    
users_profiles = pd.read_feather('./outputs/CB/users_profiles.feather')
#Load item profiles
items_profiles = pd.read_feather('./outputs/CB/items_profiles.feather')

"""
Matrix Factorization
"""
# Load the model svd and trainset from the file
algo_file = './outputs/MF/svd_tunned_algo.sav'
with open(algo_file, 'rb') as file:
    svd_tunned_algo = pickle.load(file)
trainset_file = './outputs/MF/trainset.sav'
with open(trainset_file, 'rb') as file:
    trainset = pickle.load(file)


"""
Random Recommendation 
"""
# Load the model svd and trainset from the file
algo_file = './outputs/Random/random_algo.sav'
with open(algo_file, 'rb') as file:
    random_algo = pickle.load(file)


#CB - Recommen k items for a given user#
def content_based_recommend(user_name,k,dis_metric):
    #Get user_name id#
    user_id = usernames_ids_dict.get(user_name,0)
    #Get user_profile by user_id
    user_profile = ut.get_user_profile_by_user_id(users_profiles,user_id)
    #Get top_k item ids,scores and estimated rating#
    top_k_items_ids,top_items_scores,ratings = ut.get_top_k_items_for_specific_user(k,user_profile,items_profiles,dis_metric)


    #Get items names according to item_ids
    recommendation_df = ut.get_items_names(ids_items_dict,top_k_items_ids)
    recommendation_df['Metric_Score'] = top_items_scores
    recommendation_df['Est_Rating'] = ratings
    recommendation_df['Usename'] = user_name

    return recommendation_df


#MF - Recommen k items for a given user#
def matrix_factorization_recommend(user_name,k):
    #Get user_name id#
    user_id = usernames_ids_dict.get(user_name,0)

    #Get top n predictions by the fitted algo
    top_n_predictions = ut.get_top_n(user_id,trainset,svd_tunned_algo,k)
    #Get top items ids
    top_items_ids = [obj.iid for obj in top_n_predictions]
    #Get top items rating
    top_k_ratings = [obj.est for obj in top_n_predictions]

    #Get items names according to item_ids
    recommendation_df = ut.get_items_names(ids_items_dict,top_items_ids)
    recommendation_df['Est_Rating'] = top_k_ratings
    recommendation_df['Usename'] = user_name
    return recommendation_df


def recommand_item_for_user(user_name,k,rec_type,dist_metric='CB_Cosine'):
    top_items_scores=[]
    #Get user_name id#
    user_id = usernames_ids_dict.get(user_name,0)
    #Content Based Recommendation#
    if rec_type=='CB':
        #Get user_profile by user_id
        user_profile = ut.get_user_profile_by_user_id(users_profiles,user_id)
        #Get top_k item ids,scores and estimated rating#
        top_k_items_ids,top_items_scores,top_k_ratings = ut.get_top_k_items_for_specific_user(k,user_profile,items_profiles,dist_metric)
    else:
        #Matrix Factorization or Random Recommendation#
        if rec_type=='MF':
            algo=svd_tunned_algo
        else:
            algo=random_algo
        #Get top n predictions by the fitted algo
        top_n_predictions = ut.get_top_n(user_id,trainset,algo,k)
        #Get top items ids
        top_k_items_ids = [obj.iid for obj in top_n_predictions]
        #Get top items rating
        top_k_ratings = [obj.est for obj in top_n_predictions]

    return build_recommendation_dataframe(rec_type,top_k_items_ids,top_k_ratings,top_items_scores,user_name)

#Build recommendation dataframe for collected data
def build_recommendation_dataframe(rec_type,top_k_items_ids,top_k_ratings,top_items_scores,user_name):
    #Get items names according to item_ids
    recommendation_df = ut.get_items_names(ids_items_dict,top_k_items_ids)
    
    recommendation_df['Est_Rating'] = top_k_ratings
    if rec_type=='CB':
        recommendation_df['Metric_Score'] = top_items_scores
    recommendation_df['Usename'] = user_name
    return recommendation_df


