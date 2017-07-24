import csv
import json
import os
import pandas as pd
import pickle
from sklearn.externals import joblib


def get_paths():
    paths = json.loads(open("Settings.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def identity(x):
    return x

# For pandas >= 10.1 this will trigger the columns to be parsed as strings
converters = { "jobId" : identity
             , "companyId": identity
             , "jobType": identity
             , "degree": identity
             , "major": identity
             , "industry": identity
             , "yearsExperience": identity
             , "milesFromMetropolis": identity
             }

conv2 = { "jobId" : identity
             , "salary": identity
             }

"""conv9 = { "jobId" : identity
             , "companyId": identity
             , "jobType": identity
             , "degree": identity
              , "major": identity
             , "industry": identity
             , "yearsExperience": identity
             , "milesFromMetropolis": identity
             "salary": int
             }
"""
def get_train_f_df():
    train_f_path = get_paths()["train_f_data_path"]
    return pd.read_csv(train_f_path, converters=converters)

def get_train_s_df():
    train_s_path = get_paths()["train_s_data_path"]
    return pd.read_csv(train_s_path, converters=conv2)

def get_test_df():
    test_path = get_paths()["test_data_path"]
    return pd.read_csv(test_path, converters=converters)

def get_train_df():
    train_path = get_paths()["train_data_path"]
    return pd.read_csv(train_path, converters=conv9)

def get_valid_df():
    valid_path = get_paths()["valid_data_path"]
    return pd.read_csv(valid_path, converters=converters)

def save_model(model):
    out_path = get_paths()["model_path"]
    pickle.dump(model, open(out_path, "wb"))
    
def save_model2(model):
    out_path = get_paths()["model_path"]
    joblib.dump(model,open(out_path, "wb"))

def load_model():
    in_path = get_paths()["model_path"]
    return pickle.load(open(in_path))

def write_submission(predictions):
    prediction_path = get_paths()["prediction_path"]
    writer = csv.writer(open(prediction_path, "w"), lineterminator="\n")
    valid = get_valid_df()
    rows = [x for x in zip(valid["jobId"], predictions.flatten())]
    writer.writerow(("jobId", "salary"))
    writer.writerows(rows)
    
    