'''
Naive Bayes 
Authors: Christopher Mendoza, Gerson Medina Santos, Maliheh Zargaran 
Version: 1.1
'''



import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
from sklearn.feature_extraction import stop_words

sw = []
dfl = []
chris = [0,8,13]
gerson = [14]
mali = [4,5,12,19]

 
SW = stop_words.ENGLISH_STOP_WORDS
SWFile = open('C:/Users/maxim/Desktop/gersonKeywords.txt', 'r')
yourResult = [line.split() for line in SWFile.readlines()]
for g in yourResult:
    sw.append(g[0])

#for l in SW:    
#    sw.append(l)
    

TrainingData = chris + gerson

count_vect = CountVectorizer(stop_words = sw)
info = pd.DataFrame()
testdf = pd.DataFrame()
year = []
count = 0
for x in range (1998,2018):
    df = pd.read_excel('C:/Users/maxim/Desktop/data.xlsx',sheet_name = str(x))
    year.append(df)
for y in TrainingData:
    dfl.append(year[y])
result = pd.concat(dfl)
result = result.reset_index(drop = True)

rm = result['Category'].tolist()
o = []
for s in rm:
    s = s.strip()
    o.append(s)
result['Category'] = o


info['info'] = result['Heading'] + ' ' + result['Description']
cat = list(result['Category'].unique())
X_train_counts = count_vect.fit_transform(info['info'])
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)




clf = MultinomialNB().fit(X_train_tfidf, result['Category'])

testdfinit = pd.DataFrame(year[13])
testdf['info'] = testdfinit['Heading'] + ' ' + testdfinit['Description']
#docs_new = ['TCP IP UDP', 'Internet Explorer','Media Player','Handle Objects in Memory']
X_new_counts = count_vect.transform(testdf['info'])
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)


testdf['Predicted'] = predicted
testdf['True'] = testdfinit['Category']
acc = (predicted == testdfinit['Category'])
testdf['Did it work?'] = acc

for a in acc:
    if a == True:
        count = count + 1
print(count/len(acc))


