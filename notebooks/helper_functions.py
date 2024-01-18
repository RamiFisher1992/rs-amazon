import pandas as pd
import numpy as np 

#Gets a userid , algo and dataset and returns top_k items
def get_top_n(user_id,dataset,algo, n=10):
    # Get a list of all items and all users in the dataset
    items = dataset.all_items()
    # Convert raw ids to inner ids
    raw_user_id = user_id  # replace with the user ID you are interested in
    inner_user_id = dataset.to_inner_uid(raw_user_id)
    # Find items that the user has not rated yet
    test_items = [dataset.to_raw_iid(item) for item in items if not dataset.ur[inner_user_id].__contains__(item)]
    # Predict ratings for all the items not rated by the user
    predictions = [algo.predict(raw_user_id, item) for item in test_items]
    # Get the top N recommendations
    top_n_items = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    return top_n_items

#Given items predictions and returns items data in dataframe
def convect_predictions_to_items(predictions,df):    
    #Get top items ids
    top_items_ids = [obj.iid for obj in predictions]

    item_list = (df[df['item_id']==rid].iloc[0,1:4] for rid in top_items_ids)
    top_k_df = pd.DataFrame()
    for item in item_list:
        top_k_df = top_k_df.append(item, ignore_index=True)
    return top_k_df


