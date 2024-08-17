#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# In[3]:


#사고율이 10을 넘는 관측치는 이상치로 판단해 모델링 과정에서 제거
mdl_df=df[df['사고율']<=10]


# In[4]:


#사고율이 높은 고객 탐지 성능 향상을 위해 고위험군으로 의심되는 고객을 선별적으로 분류하는 이진 분류 모델링 진행
mdl_df['고위험']=0
mdl_df.loc[mdl_df['사고율']>=1,'고위험']=1


# In[5]:


mdl_df=mdl_df.drop(['유효대수','사고건수'],axis=1)
X=mdl_df.drop(['고위험','사고율'],axis=1)
y=mdl_df['고위험']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=831)


# In[6]:


X=X_train
y=y_train


# In[7]:


X.info()


# In[8]:


#One-hot Encoding
#파이썬 logistic regression은 범주형 자료에 대해 따로 One-hot Encoding을 진행해주지 않기 때문
X = pd.get_dummies(X, columns = ['직전3년간사고건수','운전자한정특별약관'], drop_first = False)


# In[9]:


#유의하지 않았던 변수 제거
X=X.drop(['차량경과년수','마일리지약정거리','자차보험_가입여부','저경력운전자','직전3년간사고건수_5','운전자한정특별약관_1','운전자한정특별약관_2','운전자한정특별약관_5','운전자한정특별약관_6','운전자한정특별약관_10'],axis=1)


# In[10]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.metrics import matthews_corrcoef


# In[11]:


model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X, y)


# In[12]:


X_test = pd.get_dummies(X_test, columns = ['직전3년간사고건수','운전자한정특별약관'], drop_first = False)
X_test=X_test.drop(['차량경과년수','마일리지약정거리','자차보험_가입여부','저경력운전자','직전3년간사고건수_5','운전자한정특별약관_1','운전자한정특별약관_2','운전자한정특별약관_5'],axis=1)


# In[13]:


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] 
fpr, tpr, thresholds = roc_curve(y_test, y_prob)


# In[14]:


roc_auc = auc(fpr, tpr)
roc_auc


# In[15]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


# In[16]:


optimal_idx=np.argmax(tpr-fpr)
optimal=thresholds[optimal_idx]

optimal


# In[17]:


#optimal threshold로 logistic regression 진행
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X,y)
threshold=optimal
y_prob=model.predict_proba(X_test)[:,1]
y_pred=(y_prob>threshold).astype(int)


# In[18]:


test_pred=pd.DataFrame(y_pred,columns=['고위험군 예측'])
idx=X_test.index
test_pred.set_index(idx,inplace=True)
result=pd.concat([X_test,y_test],axis=1)
result=pd.concat([result,test_pred],axis=1)
result.drop('고위험',axis=1,inplace=True)
result.head()


# In[19]:


result=pd.concat([result,mdl_df.loc[idx]['사고율']],axis=1)


# In[20]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
report = classification_report(y_test, y_pred)
print("분류 보고서:\n", report)


# In[24]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[21]:


f1_score(y_test,y_pred)


# In[22]:


f1_score(y_test,y_pred,average='macro')


# In[23]:


confusion = confusion_matrix(y_test,y_pred)
confusion


# In[160]:


result.to_csv('test_dataset.csv',index=False,encoding='cp949')

