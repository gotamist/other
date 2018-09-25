#!/usr/bin/python

import sys
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.chdir('/home/thojo/work/ud/ml/fraud_from_email')
sys.path.append("../tools/")
from utils import *

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'exercised_stock_options', 'long_term_incentive', 'from_this_person_to_poi', 'from_poi_to_this_person', 'shared_receipt_with_poi'] # need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL') #Obvious choice
data_dict.pop( 'THE TRAVEL AGENCY IN THE PARK' ) # a whole bunch of NaNs and clearly not a person
data_dict.pop('LOCKHART EUGENE E') # NaN in all field except poi 
data_dict.pop('BELFER ROBERT')
    
### Task 3: Create new feature(s)
df = pd.DataFrame.from_records(list(data_dict.values()), index=data_dict.keys())  
email_df = df['email_address']

df = clean_and_move_up_poi(df) 
  
for colname in df.columns.values:
    df[ colname ] = pd.to_numeric( df[ colname ], errors = 'coerce' ) 
df[ 'emails_to_poi_ratio' ] = df[ 'from_this_person_to_poi' ] /   df[ 'from_messages' ] #every single email from 'HUMPHREY GENE E' was to a poi
df[ 'emails_from_poi_ratio' ] = df[ 'from_poi_to_this_person' ] /  df[ 'to_messages' ] 
df[ 'poi_communication' ] = df[ 'emails_to_poi_ratio' ] + df[ 'emails_from_poi_ratio' ] #collinear with the other two features created just above
df[ 'shared_receipt_ratio' ] =  df[ 'shared_receipt_with_poi' ] /  df[ 'to_messages' ] 
# compensation features
# does a poi tend to exercise options more?
# note that the numberator and denominator are not of the same kind (num. is #of options and den. is in dollars), but feature_scaling (cross-sectional) will take care of that later
df[ 'non_salary_ratio' ] = ( df[ 'bonus' ] + df[ 'long_term_incentive']  ) / df[ 'total_payments' ] # do non-pois get paid mainly in salary?
df[ 'option_exercise_ratio' ] =  df[ 'exercised_stock_options' ] / df[ 'total_stock_value' ] #ignore the negative total stock value of ROBERT BELFER.  Go back up and delete that record

#df1 = scale_features(df, [ 'option_exercise_ratio' ] ) # no need to do this.

### Store to my_dataset for easy export below.
my_dataset = df.to_dict('index')

### Extract features and labels from dataset for local testing
f_list = ['poi','salary', 'bonus', 'exercised_stock_options', 'long_term_incentive', 'from_this_person_to_poi', #leaving out to_oi as that is represented in the ratio later in this list
          'shared_receipt_ratio', 'emails_to_poi_ratio', #leaving out emails_from_poi due to collinearlity
          'poi_communication','non_salary_ratio',  'option_exercise_ratio'          
          ] # 

scale_list = [ ]
data = featureFormat(my_dataset, f_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)