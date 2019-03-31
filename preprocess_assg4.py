import pandas as pd
import numpy as np
import random
import math

global_encoder_by_field = {}
def get_encoding(column):
    column = column.astype('category')
    encoding = {}
    for i, category in enumerate(column.cat.categories):
        encoding[category] = i
    global_encoder_by_field[column.name] = encoding
    return column.cat.codes

def split_dataset(t_frac, random_state, dataset):
    '''
    split dataset
    '''
    testset=dataset.sample(frac=t_frac,random_state=random_state)
    trainset=dataset.drop(testset.index)
    testset.to_csv("testSet.csv", index = False)
    trainset.to_csv("trainingSet.csv", index = False)

def preprocess(filename):
    dating = pd.read_csv(filename)
    dating = dating.head(6500)
    cols_to_delete = ['race','race_o','field']
    for col in cols_to_delete:
        del dating[col]

    dating[['gender']] = dating[['gender']].apply(get_encoding)

    partner_cols = ['pref_o_attractive','pref_o_sincere','pref_o_intelligence','pref_o_funny','pref_o_ambitious','pref_o_shared_interests']
    participant_cols = ['attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important']  

    total_partner = 0
    total_participant = 0 

    for i in range (0,6):
        total_partner += dating[partner_cols[i]]
        total_participant += dating[participant_cols[i]] 

    for i in range(0,6):
        dating[partner_cols[i]]/=total_partner
        dating[participant_cols[i]]/=total_participant

    for i in range(0,6):
        participant_mean = dating[participant_cols[i]].sum()/len(dating[participant_cols[i]])
    #     print ('Mean of ', participant_cols[i], ':', round(participant_mean, 2))
    for i in range(0,6): 
        partner_mean = dating[partner_cols[i]].sum()/len(dating[partner_cols[i]])
    #     print ('Mean of ', partner_cols[i], ':', round(partner_mean, 2))

    non_binned_cols = ['gender', 'race', 'race_o', 'samerace', 'field', 'decision']   
    for column in dating:
        if column not in non_binned_cols:
            dating[column] = pd.cut(dating[column], 2, labels = [0,1])

    split_dataset(0.2, 47, dating)

def main():
    filename = 'dating-full.csv'
    preprocess(filename)

if __name__ == '__main__':
    main()