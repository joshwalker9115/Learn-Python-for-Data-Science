import numpy as np
from lightfm.datasets import stackexchange #fetch_movielens - kept for reference
from lightfm import LightFM
from lightfm import evaluation
from Lib import re

#original model abandoned to keep code length shorter/ prevent causing errors with sample_reccomendation()

#fetch data and format it
#data = fetch_movielens(min_rating=4.0)
data2 = stackexchange.fetch_stackexchange('crossvalidated', test_set_fraction=0.2, min_training_interactions=1, data_home=None, indicator_features=True, tag_features=False, download_if_missing=True)

#print training and testing data
#print(repr(data['train']))
#print(repr(data['test']))

#create models
#model = LightFM(loss='warp')
model1 = LightFM(loss='logistic')
model2 = LightFM(loss='bpr')
model3 = LightFM(loss='warp-kos')

#list models
models = [model1, model2, model3]

#train models
#model.fit(data['train'], epochs=30, num_threads=2)
#Second dataset
model1.fit(data2['train'], epochs=30, num_threads=2)
model2.fit(data2['train'], epochs=30, num_threads=2)
model3.fit(data2['train'], epochs=30, num_threads=2)

def sample_reccomendation(model, data, user_ids):

    #number of users and movies in training data
    n_users, n_items = data['train'].shape

    #generate reccomendations for each user we input
    for user_id in user_ids:
        #movies they already like
        known_positives = data['item_feature_labels'][data['train'].tocsr()[user_id].indices]
        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        #rank them in order of most liked to least
        top_items = data['item_feature_labels'][np.argsort(-scores)]
        #print out the results
        print("User %s" % user_id)

        print("     Known positives:")
        for x in known_positives[:3]:
            print("          https://stats.stackexchange.com/questions/%s" % re.sub('question_id:', '', x))
        
        print("     Reccommended")
        for x in top_items[:3]:
            print("          https://stats.stackexchange.com/questions/%s" % re.sub('question_id:', '', x))

def best_reccomendation():

    #define variables
    best = 0.0
    best_model = ''

    for model in models:
        score = 0.0
        pak_score = evaluation.precision_at_k(model, data2['test'])
        score += np.mean(pak_score)

        rak_score = evaluation.recall_at_k(model, data2['test'])
        score += np.mean(rak_score)

        auc_score = evaluation.auc_score(model, data2['test'])
        score += np.mean(auc_score)

        rr_score = evaluation.reciprocal_rank(model, data2['test'])
        score += np.mean(rr_score)

        print(score)
        if score >= best:
            best = score
            best_model = model
    
    return best_model

best_model = best_reccomendation()
print(best_model)
sample_reccomendation(best_model, data2, [3, 24, 40])