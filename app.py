from flask import Flask, render_template, request
from src.recommendation import recommand_item_for_user
import json


app = Flask(__name__)


# Load dictionary as a JSON file
with open('./outputs/IDs/usernames_ids_dict.json', 'r') as file:
    usernames_ids_dict = json.load(file)
    usernames=[*usernames_ids_dict]

rec_methods = [{'metric': 'CB_Cosine'},{'metric': 'CB_Euclidean'}, {'metric': 'MF_SVD'}, {'metric': 'Random'}]


@app.route("/")
def home():
    return render_template("index.html",
    rec_methods=rec_methods,usernames=usernames)


@app.route('/recommend',methods=['GET', 'POST'])
def recommend():
    
    input_data = list(request.form.values())
    user_name = input_data[0]
    rec_method = input_data[1]
    top_k = min(int(input_data[2]),10)
    
    if rec_method == 'CB_Cosine' or rec_method == 'CB_Euclidean':
        top_k_items = recommand_item_for_user(user_name,int(top_k),'CB',rec_method)
        #top_k_items = content_based_recommend(user_name,int(top_k),dis_metric) 
    elif rec_method=='MF_SVD':
        top_k_items = recommand_item_for_user(user_name,int(top_k),'MF')
        #top_k_items = matrix_factorization_recommend(user_name,int(top_k))
    else:
        top_k_items = recommand_item_for_user(user_name,int(top_k),'Random')

    top_k_items.reset_index(drop=True,inplace=True)
    return render_template('index.html',
                            rec_methods=rec_methods,
                            usernames=usernames,
                            top_k_html=top_k_items.to_html())



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)