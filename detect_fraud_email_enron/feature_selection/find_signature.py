#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)

''' Vectorize and then use feature_importances_ from prelimary decision tree 
fits to the email data to see if there are some stray words that would readily
identify the author of this email (in this case, the author's email address is 
present and that's a give away).  Extraordinarily high performance on the test 
set is the give-away. Then add those terms to the signature removal in 
vectorize_text.py '''

### The words (features) and authors (labels), already largely processed.
### These files should have been created previously

words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### Use decision tree classifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit( features_train, labels_train)
print clf.score( features_test, labels_test ) # 94.82%, 82.1% after Sara's and Chris's email addresses where removed
#print clf.score( features_train, labels_train) #100%, check

imp = clf.feature_importances_
imp_feature_idx=np.where (imp >  0.2) #14343
print vectorizer.get_feature_names()[imp_feature_idx[0][0]]


#import matplotlib.pyplot as plt
#plt.hist(imp,bins=100)


