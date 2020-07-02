# SMS SPAM Detector - Using Python 

-Develop an SMS spam detector using Naive Bayes. We will predict whether an SMS text is spam or not.(Classification Problem) 

##  Dataset -
Dataset can be downloaded from :- https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
OR Dataset is also available in ./data folder

## Libraries Used -

#### For reading data from file
import pandas as pd

#### For splitting the data into training and test dataset
from sklearn.model_selection import train_test_split

#### For creating BOW(Bag of words)
from sklearn.feature_extraction.text import CountVectorizer

#### Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB

#### Mertics for testing our models prediction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


## Directory structure

 ----
 
	|
	|--Machiene Learning Projects------------
											|						
											|---- SMS Spam Detector----------                                            
																			|			
																			|----data-----SMSSpamCollection  
																			|
																			|----necessary.py
																			|
																			|----sms_spam_detector.py
																			|
																			|----Readme