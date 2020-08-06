#import necessary modules
import necessary

#Load the dataset
current_dir = os.getcwd()      #current working directory

words_file = os.path.join(current_dir,"data\\word_data.pkl")
authors_file = os.path.join(current_dir,"data\\email_authors.pkl")

#Read author data files
authors_file_handler = open(authors_file,"rb")
authors = pickle.load(authors_file_handler)
authors_file_handler.close()

#Read word data files
words_file_handler = open(words_file,"rb")
words = pickle.load(words_file_handler)
words_file_handler.close()

# Split data into train and test sets

# using train_test_split 
X_train,X_test,y_train,y_test = train_test_split(words,authors,
                                                test_size = 0.1,
                                                random_state = 10)


# Text vectorization--go from strings to lists of numbers

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# Reduce the amount of features because the text has too many features (words)
#convert the dataset to np array 
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(features_train, labels_train)
X_train = selector.transform(X_train).toarray()
X_test = selector.transform(X_test).toarray()


# Info on the data
#Since Chris emails are denoted by 1 in training set and Sara emails are denoted by 0
#As a result sum(y_train) will give the no. of Chris training emails
#And Sara emails = len(y_train) - Chris emails
print("\nNo. of Chris training emails:", sum(y_train))
print("No. of Sara training emails:", len(y_train)-sum(y_train))


# Train the NB model

t0 = time()
model = GaussianNB()
model.fit(X_train, y_train)
print(f"\nTraining time: {round(time()-t0, 3)}s")
t0 = time()
score_train = model.score(X_train, y_train)
print(f"Prediction time (train): {round(time()-t0, 3)}s")
t0 = time()
score_test = model.score(X_test,y_test)
print(f"Prediction time (test): {round(time()-t0, 3)}s")
print("\nTrain set score:", score_train)
print("Test set score:", score_test)
