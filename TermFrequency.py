'''
Created on Aug 28, 2015

@author: Mohammadreza
'''
import numpy as np
import pandas as pd
import os 
import sklearn.feature_extraction.text as skltext
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def sortme(item):
    return item[1]

data=pd.read_csv(os.getcwd()+'\product-export\export_HK.csv')
d=data.replace(np.nan,'',regex=True)
# print data['Description'][0]
# newstr=["".join(d['Description'])]
# print len(newstr[0])


vectorizer=skltext.CountVectorizer(min_df=1,token_pattern=r"\b[^\W\d_]{3,}\w*\b",stop_words='english',ngram_range=(1,2))
X=vectorizer.fit_transform(d['Description'])
# print len(vectorizer.get_feature_names())
terms=np.array(vectorizer.get_feature_names())
frequency=np.asarray(X.sum(axis=0)).ravel()

final=zip(terms,frequency)

sorted_result=sorted(final,key=sortme,reverse=True)

# print sorted_result[1:1000]
finaldata=[]
for item in sorted_result:
    temp=[item[0],item[1]]
    finaldata.append(temp)
df=pd.DataFrame(finaldata,columns=['word','frequency'])
df.to_csv(os.getcwd()+'\HK_Procutlist.csv',index=False)
    
# mycloud=WordCloud().fit_words(sorted_result)
# plt.imshow(mycloud)
# plt.show()




# print vectorizer.get_feature_names()
# Y=X.toarray()
# print len(Y.sum(axis=0))
