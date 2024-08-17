#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


# In[3]:


df=pd.read_csv('C:/Users/EunSeon/Downloads/dataset.csv')


# In[4]:


#순서형과 명목형 범주 관계에 대해 상관계수를 파악할 수 있는 방법이 없기 때문에 명목형 범주형 자료 제거
df.drop(['피보험자성별','국산구분','운전자한정특별약관','영상기록장치특약','유효대수','사고건수','마일리지약정_가입여부','자차보험_가입여부','고경력운전자','저경력운전자','사고율'],axis=1,inplace=True)


# In[5]:


#기타를 의미하는 범주 제거
df=df[df['직전3년간사고건수']!=5]


# In[6]:


df.corr(method='spearman',min_periods=1)


# In[7]:


plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
sns.heatmap(data=df.corr(method='spearman'),annot=True,linewidths=.5,cmap='Blues')

