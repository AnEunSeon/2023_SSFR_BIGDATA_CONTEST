#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


df=pd.concat([df1,df2,df3],axis=0)


# In[5]:


#분석에 의미 없는 변수(열) 제거 
df=df.drop(['ZCPRLCLCD','ZDRVLISCD___T'],axis=1)


# In[6]:


#유효대수가 0인 관측치 제거 (사고율 계산이 불가하기 때문)
df=df[df['YUHO']!=0]


# In[7]:


#결측치 확인
df.isnull().sum()


# In[10]:


#결측치 제거
df=df.dropna(subset=['ZDPRODSCD'])


# In[11]:


#피보험자연령대가 0~10, 90~100대인 관측치 제거 (이상치로 판단)
df=df[(df['ZINSRDAVL']!=0)&(df['ZINSRDAVL']!=10)&(df['ZINSRDAVL']!=90)&(df['ZINSRDAVL']!=100)]


# In[12]:


#변수설명테이블에 정의되지 않은 값을 가진 관측치 제거 
df=df[df['NCR']!='0']


# In[13]:


df.info()


# In[14]:


df.head()


# In[15]:


#순서형 범주형 자료에 대한 Label Encoding 진행
ord_map1={20:2,30:3,40:4,50:5,60:6,70:7,80:8}
df['ZINSRDAVL']=df['ZINSRDAVL'].map(ord_map1)

ord_map2={'소형A':1,'소형B':2,'중형':3,'대형':4,'다목적1종':5,'다목적2종':5}
df['차종']=df['차종'].map(ord_map2)

ord_map3={'미가입':1,'5천만원이하':2,'1억이하':3,'1억이상':4}
df['ZCARISDAM']=df['ZCARISDAM'].map(ord_map3)

ord_map4={'미가입':1,'3000K':2,'5000K':3,'7000K':4,'10000K':5,'12000K':6,'15000K':7}
df['마일리지약정거리']=df['마일리지약정거리'].map(ord_map4)

ord_map5={'신차':1,'5년이하':2,'10년이하':3,'10년이상':4}
df['ZCARPSGVL']=df['ZCARPSGVL'].map(ord_map5)


# In[16]:


#이진형 범주형 자료에 대한 Encoding 진행 
#기존 데이터에서 범위가 (1,2)였던 것을 (0,1)으로 변경
df['ZIOSEXCD']=df['ZIOSEXCD']-1
df['ZDPRODSCD']=df['ZDPRODSCD']-1
bin_map={'미가입':0,'가입':1}
df['ZIMAGERVL']=df['ZIMAGERVL'].map(bin_map)


# In[17]:


#가독성을 위해 직전3년사고건수의 값을 mapping
ncr_map={'N':1,'D':2,'C':3,'B':4,'Z':5}
df['NCR']=df['NCR'].map(ncr_map)


# In[18]:


df['ZDPRODSCD']=df['ZDPRODSCD'].astype(int)


# In[19]:


df.head()


# In[20]:


#파생변수 생성
df['마일리지할인율_가입여부']=0
df.loc[df['마일리지약정거리']!=1,'마일리지할인율_가입여부']=1

df['자차보험_가입여부']=0
df.loc[df['ZCARISDAM']!=1,'자차보험_가입여부']=1

df['고경력운전자']=0
df.loc[df['ZCARISDAM']==8,'고경력운전자']=1

df['저경력운전자']=0
df.loc[df['ZCARISDAM']==1,'저경력운전자']=1


# In[21]:


#모델링의 종속변수로 사용할 사고율 계산
df['사고율']=df['SAGO']/df['YUHO']


# In[22]:


df.head()


# In[23]:


#편의를 위해 Column명 변경
df.columns=['피보험자연령대','피보험자성별','국산구분','직전3년간사고건수','차량경과년수','차종','운전자한정특별약관','가입경력','차량가입금액','영상기록장치특약','마일리지약정거리','유효대수','사고건수','마일리지약정_가입여부','자차보험_가입여부','고경력운전자','저경력운전자','사고율']


# In[26]:


df.info()


# In[23]:


df.to_csv('dataset.csv',index=False,encoding='cp949')

