from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pandas as pd
import pickle

train_data = pd.read_csv(r'loan_train_data.csv')
train_data = train_data.iloc[:,1:]
train_data["EMI_log"] = ((np.exp(train_data["EMI_log"]))/100)*2
target = pd.read_csv(r'loan_target.csv')
target = target.iloc[:,1:]
model = LogisticRegression(C=0.01, class_weight=0.01, solver='newton-cg')
model = ExtraTreesClassifier(max_features=5, min_samples_leaf=20, min_samples_split=30, n_estimators=300)
model.fit(train_data, target)
pickle.dump(model, open('model.pkl', 'wb'))
