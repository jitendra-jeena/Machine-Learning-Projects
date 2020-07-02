#import necessary modules
import necessary
#Load the dataset
#Dataset is available on https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
#Since the content of the file are tab separated
#so use read_table() method

dataset = pd.read_table("SMSSpamCollection",header = None,names = ["label","sms_message"])

#Here we have two columns:- Label consisting of spam or ham
#:- sms_message : the body of the message

#View first 5 rows of our dataset

print(dataset.head())
#Data Preprocessing:
#Since we are using Sklearn package 
#So we have to convert the categorical data(labels) into numerical
#One way of achieving this is by using  map() method

dataset["label"] = dataset.label.map({"ham":0,"spam":1})

#Lets view the changes is our dataset

print(dataset.head())


#Splitting the dataset into training and testing data
#Using sklearn package
X_train,X_test,y_train,y_test = train_test_split(dataset["sms_message"],
                                                dataset["label"],
                                                 random_state = 1)

#Displaying the size of our training and testing dataset
print('Number of rows in the total set: {}'.format(dataset.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

#Transforming the sms_message column using BAW (Bag of Word) 
#Create an Object of countVectorizer class

countVector = CountVectorizer(lowercase = False,stop_words = "english")
#by setting lowercase = False we are making the content case insentitive
#by setting stop_words = "engish" we are ignoring commonling occuring english word in a sentence 
#like a,an,the, he ,she,it,etc.

#Apling BAW to training dataset
training_data = countVector.fit_transform(X_train)

#Transform testing data and return the matrix
testing_data = countVector.transform(X_test)


#Fit the model
classifier = MultinomialNB()
classifier.fit(training_data,y_train)

#Make Predictions
prediction = classifier.predict(testing_data)

#Finding the accuracy,Recall score,precision and F1 score

print('Accuracy score: ', format(accuracy_score(y_test, prediction)))
# 0.9921033740129217

print('Precision score: ', format(precision_score(y_test, prediction)))
#0.9887640449438202

print('Recall score: ', format(recall_score(y_test, prediction)))
# 0.9513513513513514

print('F1 score: ', format(f1_score(y_test, prediction)))
#0.9696969696969697