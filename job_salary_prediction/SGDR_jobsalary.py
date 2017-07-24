# -*- coding: utf-8 -*-
"""
@author: yue wu
Random Forest Regression
"""
import data_io
from features import FeatureMapper, SimpleTransform
import pandas as pd
import pickle
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import explained_variance_score

def feature_extractor():
    features = [('jobType-Bag of Words', 'jobType', CountVectorizer(max_features=100)),
                ('degree-Bag of Words', 'degree', CountVectorizer(max_features=100)),
                ('major-Bag of Words', 'major', CountVectorizer(max_features=100)),
                ('industry-Bag of Words', 'industry', CountVectorizer(max_features=100)),
                ('yearsExperience-Bag of Words', 'yearsExperience', CountVectorizer(max_features=100)),
                ('milesFromMetropolis-Bag of Words', 'milesFromMetropolis', CountVectorizer(max_features=100))]
    combined = FeatureMapper(features)
    return combined

def get_pipeline():
    features = feature_extractor()
    steps = [("extract_features", features),
             ("classify", linear_model.SGDRegressor(alpha=0.000001, 
                                                 average=False, 
                                                 epsilon=0.1, 
                                                 eta0=0.01,
                                                 fit_intercept=True, 
                                                 l1_ratio=0.15, 
                                                 learning_rate='invscaling',
                                                 loss='squared_loss', 
                                                 n_iter=100, 
                                                 penalty='l2', 
                                                 power_t=0.25,
                                                 random_state=None, 
                                                 shuffle=True, verbose=0, 
                                                 warm_start=False))]
    return Pipeline(steps)

def main():

    print("Reading in the raw data of features and salaries for merging")
    train_f = data_io.get_train_f_df()
    train_s = data_io.get_train_s_df()
    #train_f: training feature data; train_s: training salary data with 0 items deleted 

    """
    train_f.describe
    train_s.describe
    """
    #merge the data by jobId, similar to SQL join
    data = pd.merge(train_f,train_s,how='left')
    data.to_csv("D:/job/indeed_data_science_exercise/RFC1/train9merge2.csv", sep=',',encoding='utf-8')

    # seperate the data into features set of the feature columns and the set with target column salary only
    #'companyId' excluded
    characters = ["jobType", "degree", "major", "industry", "yearsExperience", "milesFromMetropolis"]
    x = data[characters]
    y = data[['salary']]
         
    print("Extracting features and training model")
    classifier = get_pipeline()
    classifier.fit(xtr, ytr)

    print("Saving the classifier")
    data_io.save_model(classifier)
    
    print("Load testing data") 
    testin = data_io.get_test_df()
    test=testin[characters]

    print("Making predictions") 
    predictions = classifier.predict(test)   
    predictions = predictions.reshape(len(predictions), 1)

    #classifier.get_params
    #pred_score=explained_variance_score(ycv, predictions, multioutput='raw_values')

    print("Writing predictions to file")
    write_submission(predictions)
    
if __name__=="__main__":
    main()
