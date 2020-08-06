#For reading the data
import pickle

from time import time

#For creating train test split
from sklearn.model_selection import train_test_split

#For calculating the frequency of each word and
#reduce the weightage of most common words
from sklearn.feature_extraction.text import TfidfVectorizer

#For reducing the numbers of features
from sklearn.feature_selection import SelectPercentile, f_classif

#For building the classifier
from sklearn.naive_bayes import GaussianNB