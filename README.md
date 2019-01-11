# Traditional machine learning

This repository contains a few machine learning projects, that do not
have a deep-learning component. Decision Trees, SVMs, Naïve Bayes,
SVMs, dimensionality reduction, clustering (Guassian Mixture models,
k-means).

Other elements in the thought process, such as exploratory data
analysis, cross validation, feature importance, data normalization, appropriate
visualization, choosing the complexity of the model, evaluation of the
model are considered

## Ensemble methods

HereThis is a classification problem on the [census income data
set](https://archive.ics.uci.edu/ml/datasets/Census+Income) from the
UCI Machine Learning repository.  The problem is to look at various
characteristics of a person such as the number of years of formal
education, age, gender, profession *etc*, to try and classsify their
income as being below or above $50,000.

After preprocessing of the data, various supervised models (DT, SVM,
kNN) are trained on it.  Then, I pick one of them, a simple decision
tree model, and use it as a weak-learning base for ensemble methods
like AdaBoost.  I then compare with random forests as well.  Feature
importance is considered and it can be seen that by if chosen
carefully, a small feature subset can be enough to get almost as good
a performance as in the case of having the full feature set.

## Boston Housing

This famous dataset contains housing data collected back in 1978.  It
has the prices of a small number of houses - only 506.  There are 14
features available for each house.  The problem is a regression
problem.  Is it possible to use features such as the number of rooms,
the neighborhood in which it is located.  The aim is learn what the
selling price would be, using these features.

The problem here is an exploration of decision tree regressors, with
all their limitations and strengths, a look at bias-variance trade-off
and the use of cross-validation.

## Clustering

### Image compression
This directory has a self-contained script to
compress any image to a few colors without losing a lot of the
information.  We should expect to be able to compress the image to
about a third of its size by this alone.  This is done by k-means
clustering the pixels in color-space.

### Identifying customer segments

Here, we segment the costumers (retailers) of a goods distribution
company based on the orders they place for various items.  Other
matters explored are EDA, PCA for dimensionality reduction.

### Simple movie recommendations

Here, the dataset is the [MovieLens User Ratings Dataset]
(https://grouplens.org/datasets/movielens/).  We have about 100,000
ratings for 9,125 movies from 183 users. k-means can be used to
cluster them into groups of users of movies. The Silhouette score is
used to explore the number of groups to clusters to use.

We can use the information from these clusters to predict ratings that
a particular user may give to a certain movie if that (user, movie)
pair has no rating recorded.

## Enron fraud detection

The other is a project for fraud detection by looking at emails from
the [Enron email dataset](https://www.cs.cmu.edu/~./enron/). The task
is to identify the people who were likely to have been involved in the
Enron scandal by looking at a number of features, most of the relevant
ones being related to their compensation details and the degree of
communication they had with others who were already known to have been
involved. Please see README within the directory for more.

## Machine

One of the items is simply a description of the GPU-enabled machine
to run my projects as an alternative to running them on the cloud.
