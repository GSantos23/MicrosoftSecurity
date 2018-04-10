'''
Naive Bayes 
Authors: Christopher Mendoza, Gerson Medina Santos, Maliheh Zargaran 
Version: 1.0
'''



import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


count_vect = CountVectorizer(stop_words = 'english')
info = pd.DataFrame()
year = []
for x in range (1998,2018):
    df = pd.read_excel('C:/Users/maxim/Desktop/data.xlsx',sheet_name = str(x))
    year.append(df)
df = [year[8],year[13]]
result = pd.concat(df)
info['info'] = result['Heading'] + '' + result['Description'] + result['Severity']
cat = list(result['Category'].unique())
X_train_counts = count_vect.fit_transform(info['info'])
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)




clf = MultinomialNB().fit(X_train_tfidf, result['Category'])

docs_new = ['TCP IP UDP', 'Internet Explorer','Driver']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)


for i in range(0,len(docs_new)):
    print(docs_new[i] + ' ---> ' + str(predicted[i]))

