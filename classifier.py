import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB

# read data into a pandas dataframe
df = pd.read_csv('spam.csv', usecols=range(0,2))
print(df.head())
# give proper names to columns
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
msg = df['message'].copy()
labels = df['label'].copy()

# train test split for cross validation
X_train, X_test, y_train, y_test = train_test_split(msg, labels, random_state=0)

# build token count matrix for words that appear in training data
count_vectorizer = CountVectorizer()
# normalize raw word count to lessen impact of very frequent but less important words
tf_transformer = TfidfTransformer()

# performing count vectorization and Tfidf Transformation on Training and test data
X_train_count = count_vectorizer.fit_transform(X_train)
X_train_tf = tf_transformer.fit_transform(X_train_count)
X_test_count = count_vectorizer.transform(X_test)
X_test_tf = tf_transformer.transform(X_test_count)

# smoothing parameters to try
alphas = np.array([0.04, 0.03, 0.02, 0.001, 0.0001, 0.2])
MNB = MultinomialNB(alpha=0.04).fit(X_train_tf, y_train)
# Iterate over all parameters(alphas) and find one that gives best cross validation score
grid_search_cv = GridSearchCV(estimator= MNB, param_grid=dict(alpha=alphas))
grid_search_cv.fit(X_test_tf, y_test)

print("Best Score found by GridSearchCV :", grid_search_cv.best_score_)
print("Best Params found by GridSearchCV :", grid_search_cv.best_params_)

# try random
messages_new = ['Please call. You just won a $trillion lottery!!!', 'The quick brown fox jumps over the lazy do', 'Lets meet for dinner tomorrow at 8']
docs_new_count = count_vectorizer.transform(messages_new)
docs_new_tf = tf_transformer.transform(docs_new_count)

predicted = MNB.predict(docs_new_tf)

print('Train Set Accuracy :{:.2f}'.format(MNB.score(X_train_tf, y_train)))
print('Test Set Accuracy :{:.2f}'.format(MNB.score(X_test_tf, y_test)))

print(predicted)

