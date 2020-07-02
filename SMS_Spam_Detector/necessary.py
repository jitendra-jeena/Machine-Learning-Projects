#For reading data from file
import pandas as pd		

#For splitting the data into training and test dataset
from sklearn.model_selection import train_test_split			

#For creating BOW(Bag of words)
from sklearn.feature_extraction.text import CountVectorizer		

#Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB			

#Mertics for testing our models prediction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score		