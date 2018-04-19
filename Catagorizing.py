'''
Categorizing Bulletins
Authors: Christopher Mendoza, Gerson Medina Santos, Maliheh Zargaran 
Version: 1.1
'''

import pandas as pd
from sklearn.feature_extraction import stop_words

sw = []
dfl = []
chris = [0,8,13]
gerson = [14]
mali = [4,5,12,19]
P = {}


#Set up stopword list 
SW = stop_words.ENGLISH_STOP_WORDS
SWFile = open('C:/Users/maxim/Desktop/gersonKeywords.txt', 'r')
yourResult = [line.split() for line in SWFile.readlines()]
for g in yourResult:
    sw.append(g[0])
    
SWFile.close()

#Pick traning data years (1998 = 0,...,2017 = 19)
TrainingData = 0,19

#Make a dataframe that will contain all the years
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



#Get unique category list
rm = result['Category'].tolist()
o = []
for s in rm:
    s = s.strip()
    o.append(s)
result['Category'] = o
cat = list(result['Category'].unique())


#Combine the Heading and Description into one body of text then remove stopwords and other common symbols
info['info'] = result['Heading'] + ' ' + result['Description']
info['Category'] = result['Category']
info['info'] = info['info'].str.lower()
info['info'] = info['info'].apply(lambda x: ' '.join([word for word in x.split() if word not in (sw)]))
info['info'] = info['info'].str.replace('"','')
info['info'] = info['info'].str.replace("'",'')
info['info'] = info['info'].str.replace('(','')
info['info'] = info['info'].str.replace(')','')
info['info'] = info['info'].str.replace('.','')
info['info'] = info['info'].str.replace(',','')
info['info'] = info['info'].str.replace('$','')

#Get list of unique words in entire data set.
allwords = pd.DataFrame()
tempaw = pd.DataFrame()
ll = []
aw = list(info['info'].str.lower().str.split())
for d in range(0,len(aw)):
    tempaw = pd.DataFrame()
    tempaw['words'] = list(aw[d])
    ll.append(tempaw)
allwords = pd.concat(ll)
uwlist = list(allwords['words'].unique())
uwlist.sort()
WCDataSet = len(uwlist)

#Get total number of words in each category
WC = {}
for c in cat:
    pl = []
    temp = pd.DataFrame()
    words = pd.DataFrame()
    temp = info[(info['Category'] == c)]
    lp = list(temp['info'].str.lower().str.split())
    for w in range(0,len(lp)):
        for q in lp[w]:
            pl.append(q)
    words['words'] = pl
    WC[c] = len(words)
    
#Make dictionary that contains probability of each word being in every category
for word in uwlist:
    P[word] = 0
    T = {}
    for c in cat:
        pl = []
        temp = pd.DataFrame()
        words = pd.DataFrame()
        temp = info[(info['Category'] == c)]
        lp = list(temp['info'].str.lower().str.split())
        for w in range(0,len(lp)):
            for q in lp[w]:
                pl.append(q)
        words['words'] = pl
        words = words[(words['words'] == word)]
        prob = (len(words) + 1) / (WC[c] + WCDataSet)
        T[c] = prob
    P[word] = T
    
Result = {}


#Make dictionary that contains the probability for each word for each category
for c in cat:
    Result[c] = 0
    T = {}
    for word in uwlist:
        T[word] = P[word][c]
    Result[c] = T
    
        
    
    
    