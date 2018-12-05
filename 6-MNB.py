from sklearn.datasets import fetch_20newsgroups

abc=fetch_20newsgroups()
print(abc["target_names"])

#CASS
categories=['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
news_train=fetch_20newsgroups(subset='train',categories=categories,shuffle='true')
news_test=fetch_20newsgroups(subset='test',categories=categories,shuffle='true')
print("Target Names",news_train.target_names)

# PTM - Pipeline, TfidfVectorizer, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
text_clf=Pipeline([('vect',TfidfVectorizer()),('clf',MultinomialNB())])
text_clf.fit(news_train.data,news_train.target) # Train's Data & Target are "Fit"ted.
predicted=text_clf.predict(news_test.data) # Test's Data is "Predict"ed.

# Metrics
from sklearn import metrics
 # AccuracyScore = Test's Target + Predicted
print("accuracy",metrics.accuracy_score(news_test.target,predicted))
# ClassificationReport = Test's Target + Predicted + Test's Target Names
print(metrics.classification_report(news_test.target,predicted,target_names=news_test.target_names)) 
# ConfusionMatrix = Test's Target + Predicted
print("Confusion matrix:\n",metrics.confusion_matrix(news_test.target,predicted))