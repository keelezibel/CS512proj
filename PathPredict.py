import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import scipy.optimize as op
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from scipy.optimize import minimize
from numpy import dot
from __future__ import division
from sklearn.metrics import roc_auc_score

os.chdir('E:\\Stat\\cs 512\\final project')


mytrain = np.genfromtxt('train_top_author.txt',delimiter='\t',dtype=str,skip_header=1)
mytest = np.genfromtxt('test_top_author.txt',delimiter='\t',dtype=str,skip_header=1)

def column(matrix, i):
  return [row[i] for row in matrix]

repo_name=list(set(column(mytrain,0)))
repo_owner=list(set(column(mytrain,1)))
actor=list(set(column(mytrain,2)))
language=list(set(column(mytrain,3)))
type=list(set(column(mytrain,4)))

repo_name_test=list(set(column(mytest,0)))
repo_owner_test=list(set(column(mytest,1)))
actor_test=list(set(column(mytest,2)))
language_test=list(set(column(mytest,3)))
type_test=list(set(column(mytest,4)))

#Generate Response Feature according to ARA#
def co_actor(a,b,j):
 output=[]
 for i in range(len(a)):
  c=[]
  c=b[b[:,j]==a[i]]
  output.extend(c)
 return output;

def generate_ara(target1,target2):
 path=[]
 for i in range(len(target1)):
  share=[]
  share=target2[target2[:,2]==target1[i]]
  share_item=[]
  share_item=list(set(column(share,0)))
  share_actor=[]
  share_actor=co_actor(share_item,target2,0)
  co_acotr1=[]
  co_actor1=list(set(column(share_actor,2)))
  for j in range(len(co_actor1)):
   path.append([target1[i],co_actor1[j]])
 return path
 
ara=generate_ara(actor,mytrain)
ara_test=generate_ara(actor_test,mytest)  
np.savetxt('ARA.txt', ara, delimiter=' ',fmt="%s")
np.savetxt('ARA_Test.txt', ara_test, delimiter=' ',fmt="%s")

def generate_pairs(target,j):
 pairs=[]
 for i in range(len(target)):
  pairs.append([target[j],target[i]])
 pairs.pop(j)
 return pairs

example=generate_pairs(actor,1644)
example1=generate_pairs(actor_test,1644)
 
def ara_check(target):
 ara_path=np.zeros(shape=(len(target),1))
 for i in range(len(target)):
  if target[i] in ara:
   ara_path[i]=1
 return ara_path

def ara_check_test(target):
 ara_path=np.zeros(shape=(len(target),1))
 for i in range(len(target)):
  if target[i] in ara_test:
   ara_path[i]=1
 return ara_path


p=ara_check(example)
p1=ara_check_test(example1)

def intersect(a, b):
    return list(set(a) & set(b))

#Generate Meta-Path Count as features#
def generate_path(target1,target2,j,k):
 patha=[]
 pathb=[]
 itema=[]
 itemb=[]
 intersect1=[]
 patha=target2[target2[:,2]==target1[j][0]]
 pathb=target2[target2[:,2]==target1[j][1]]
 itema=list(set(column(patha,k)))
 itemb=list(set(column(pathb,k)))
 intersect1=intersect(itema,itemb)
 sum_count=0
 if len(intersect1)!=0:
  count=np.zeros(shape=(len(intersect1),1))
  for q in range(len(intersect1)):
   pathc=[]
   pathd=[]
   subset=[]
   itemc=[]
   itemd=[]
   subset=target2[target2[:,k]==intersect1[q]]
   pathc=subset[subset[:,2]==target1[j][0]]
   pathd=subset[subset[:,2]==target1[j][1]]
   itemc=list(set(column(pathc,0)))
   itemd=list(set(column(pathd,0)))
   count[q]=len(itemc)*len(itemd)
   sum_count=sum_count+count[q]
 return sum_count

def feature_path(target1,target2,k):
 meta_path=np.zeros(shape=(len(target1),1))
 for i in range(len(meta_path)):
  meta_path[i]=generate_path(target1,target2,i,k)
 return meta_path

arora=feature_path(example,mytrain,1)
arlra=feature_path(example,mytrain,3)
arera=feature_path(example,mytrain,4)
intern=np.ones(shape=(2999,1))

train_path=np.concatenate((intern,arora,arlra,arera), axis=1)

arora_test=feature_path(example1,mytest,1)
arlra_test=feature_path(example1,mytest,3)
arera_test=feature_path(example1,mytest,4)
intern_test=np.ones(shape=(2999,1))

test_path=np.concatenate((intern_test,arora_test,arlra_test,arera_test), axis=1)

#Logistic Regression#
model = LogisticRegression()
p = np.ravel(p)
model = model.fit(train_path, p)

pd.DataFrame(zip(train_path.columns, np.transpose(model.coef_)))

predicted = model.predict(test_path)
probs = model.predict_proba(test_path)
print metrics.accuracy_score(p1, predicted)

scores = cross_val_score(LogisticRegression(), train_path, p, scoring='accuracy', cv=10)
roc_auc_score(p1, predicted)

#Maximum Likelihood Estimation#
def lnlike(beta,arora,arlra,arera):
 b0,b1,b2,b3=beta
 logtimes=[]
 times=np.zeros(shape=(len(train_path),1))
 sum_times=0
 for i in range(len(train_path)):
  times[i]=p[i]*np.log(1/(1+np.exp(-(b0+arora[i]*b1+arlra[i]*b2+arera[i]*b3))))+(1-p[i])*np.log(1-1/(1+np.exp(-(b0+arora[i]*b1+arlra[i]*b2+arera[i]*b3))))
  sum_times=sum_times+times[i]
 logtimes=-sum_times
 return logtimes

coef=np.array(zip(np.transpose(model.coef_)))
b0_true=coef[0]
b1_true=coef[1]
b2_true=coef[2]
b3_true=coef[3]

nll = lambda *args: lnlike(*args)
result = op.minimize(nll,[b0_true,b1_true,b2_true,b3_true],args=(arora,arlra,arera))
b0_ml,b1_ml, b2_ml, b3_ml = result["x"]
