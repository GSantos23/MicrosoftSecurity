from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
#import pylab as plt
import pandas as pd
from sklearn.feature_extraction import stop_words

import numpy as np
import matplotlib.pyplot as plt

sw = []
dfl = []

#Set up stopword list 
SW = stop_words.ENGLISH_STOP_WORDS
#SWFile = open('/Users/meli/Documents/Research/Naive/codes/preprocessing/stpwords.txt', 'r')
SWFile = open('/home/g3rs0n/UTEP/hossain/stpwords.txt', 'r')
yourResult = [line.split() for line in SWFile.readlines()]
for g in yourResult:
    sw.append(g[0])
    
SWFile.close()

    
#Pick traning data years (1998 = 0,...,2017 = 19)
#TrainingData = [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,18,19]
#TrainingData = [0,1,2,3,4,5,6,7] #21.84, 21.84
#TrainingData = [0,1,2,3,4,5,6,7,8,9] #35.44
#TrainingData = [0,1,2,3,4,5,6,7,8,9,11,12] #37.86
#TrainingData = [0,1,2,3,4,5,6,7,8,9,11,12,13,14] #45.63
#TrainingData = [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16] #50.97
TrainingData = [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,18,19] #51.46

#Setup count vectorizor with stopwords
count_vect = CountVectorizer(stop_words = sw)

#Make a dataframe that will contain all the years
info = pd.DataFrame()
testdf = pd.DataFrame()
year = []
count = 0
for x in range (1998,2018):
    df = pd.read_excel('/home/g3rs0n/UTEP/hossain/mali.xlsx',sheet_name = str(x))
    #df = pd.read_excel('/Users/meli/Documents/Research/Naive/codes/preprocessing/withoutMisc.xlsx',sheet_name = str(x))
    year.append(df)
for y in TrainingData:
    dfl.append(year[y])
result = pd.concat(dfl, sort=False)
result = result.reset_index(drop = True)
#size of result in each training set is considered as num of bulletins in those years
# new concat() requires sort=false


#Get unique category list
rm = result['Category'].tolist()
o = []
for s in rm:
    s = s.strip()
    o.append(s)
result['Category'] = o
cat = list(result['Category'].unique())


#Combine the Heading and Description into one body of text
info['info'] = result['Heading'] + ' ' + result['Description']

#Vectorize the info and get the tfidf
X_train_counts = count_vect.fit_transform(info['info'])
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#Train the classifier
clf = MultinomialNB().fit(X_train_tfidf, result['Category'])

#Vectorize and transform test year
#testdfinit = pd.DataFrame(year[0]) #0.35
#testdfinit = pd.DataFrame(year[1]) #0.29508196721311475
#testdfinit = pd.DataFrame(year[2]) #0.21
#testdfinit = pd.DataFrame(year[3]) #0.3
#testdfinit = pd.DataFrame(year[4]) #0.2361111111111111
#testdfinit = pd.DataFrame(year[5]) #0.19607843137254902
#testdfinit = pd.DataFrame(year[6]) #0.26666666666666666
#testdfinit = pd.DataFrame(year[7]) #0.2545454545454545
#testdfinit = pd.DataFrame(year[8]) #0.3717948717948718
#testdfinit = pd.DataFrame(year[9]) #0.42028985507246375
#testdfinit = pd.DataFrame(year[10]) #0.44871794871794873
#testdfinit = pd.DataFrame(year[11]) #0.3918918918918919
#testdfinit = pd.DataFrame(year[12]) #0.32075471698113206
#testdfinit = pd.DataFrame(year[13]) #0.32
#testdfinit = pd.DataFrame(year[14]) #0.30120481927710846
#testdfinit = pd.DataFrame(year[15]) #0.41904761904761906
#testdfinit = pd.DataFrame(year[16]) #0.4117647058823529
#testdfinit = pd.DataFrame(year[17]) #0.4962962962962963
#testdfinit = pd.DataFrame(year[18]) #0.43137254901960786
#testdfinit = pd.DataFrame(year[19]) #0.21739130434782608

#testdfinit = pd.DataFrame(year[17])
p = [10,17]
testlist = []
for n in p:
    testlist.append(year[n])
testdfinit = pd.concat(testlist, sort=False)
#testdfinit = pd.DataFrame(year[10].join()) + pd.DataFrame(year[17]) #??????????

rm = testdfinit['Category'].tolist()
o = []
for s in rm:
    s = s.strip()
    o.append(s)
testdfinit['Category'] = o



testdf['info'] = testdfinit['Heading'] + ' ' + testdfinit['Description']
X_new_counts = count_vect.transform(testdf['info'])
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

#Classify test year with trained model
predicted = clf.predict(X_new_tfidf)

#Compare to manual results
testdf['Predicted'] = predicted
testdf['True'] = testdfinit['Category']
acc = (predicted == testdfinit['Category'])
testdf['Did it work?'] = acc

'''for a in acc:
    if a == True:
        count = count + 1
print(count/len(acc))'''

Accuracy_Score = accuracy_score(testdfinit['Category'], predicted )
Precision_Score = precision_score(testdfinit['Category'], predicted ,  average="macro")#Posetive Predicted value
Recall_Score = recall_score(testdfinit['Category'], predicted ,  average="macro")#True Posetive Rate
F1_Score = f1_score(testdfinit['Category'], predicted ,  average="macro") #F-measure 

print('\n\n')
print('Average Accuracy: %0.2f +/- (%0.1f) %%' % (Accuracy_Score.mean()*100, Accuracy_Score.std()*100))
print('Average Precision: %0.2f +/- (%0.1f) %%' % (Precision_Score.mean()*100, Precision_Score.std()*100))
print('Average Recall: %0.2f +/- (%0.1f) %%' % (Recall_Score.mean()*100, Recall_Score.std()*100))
print('Average F1-Score: %0.2f +/- (%0.1f) %%' % (F1_Score.mean()*100, F1_Score.std()*100))
print('\n\n\n')

#Confusion Matrix
labels = cat;
#testdfinit['Category'] is actual
print("Number of unique labels in ground truth: ", np.unique(testdfinit['Category']).shape)
print("Number of unique labels in prediction: ", np.unique(predicted).shape)

print("Number of test instances in ground truth: ", testdfinit['Category'].shape)
print("Number of test instances in prediction: ", predicted.shape)

CM = confusion_matrix(testdfinit['Category'], predicted )
#print(CM)

'''fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(CM)
#plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
#ax.set_xticklabels([''] + labels,fontsize=10)
#ax.set_yticklabels([''] + labels,fontsize=10)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.grid(True)
plt.show()'''

ticks=np.linspace(0, 27,num=27)
plt.imshow(CM, interpolation='none')
plt.colorbar()
#plt.xticks(ticks,fontsize=6)
#plt.yticks(ticks,fontsize=6)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.grid(True)
plt.show()



#Feature extraction:
'''from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
array = result.values
X = array[:,0:5]
Y = array[:,5]
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)'''

#number of bulletins in each category Sheet2
#TrainingData = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
numBulletinsInEachCat = result['Category'].value_counts()
#sum(numBulletinsInEachCat) #this should be eaqual to numBulletinsInEachCat

