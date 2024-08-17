#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# In[2]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error


# In[3]:


mdl_df=pd.read_csv('C:/Users/EunSeon/Downloads/dataset_train.csv')


# In[4]:


mdl_df.info()


# In[5]:


#고위험군/저위험군 모델링을 위한 데이터셋 분리
up_df=mdl_df[mdl_df['고위험']==1]
low_df=mdl_df[mdl_df['고위험']==0]


# In[13]:


#고위험군 Ensemble Regressor fitting
up_X=up_df.drop(['고위험','사고율'],axis=1)
up_y=up_df['사고율']

dt_model1=DecisionTreeRegressor(random_state=42)
rf_model1=RandomForestRegressor(random_state=42)
gb_model1=GradientBoostingRegressor(random_state=42)

ensemble_model1=VotingRegressor(estimators=[('dt', dt_model1), ('rf', rf_model1), ('gb', gb_model1)])
ensemble_model1.fit(up_X, up_y)


# In[ ]:


#feature importance, shap value 도출을 위해 각 모델에 fitting
dt_model1.fit(up_X, up_y)
rf_model1.fit(up_X, up_y)
gb_model1.fit(up_X, up_y)


# In[10]:


#feature importance
dt_feature_importance = dt_model1.feature_importances_
rf_feature_importance = rf_model1.feature_importances_
gb_feature_importance = gb_model1.feature_importances_

ensemble_feature_importance = (dt_feature_importance + rf_feature_importance + gb_feature_importance) / 3


# In[ ]:


import matplotlib.pyplot as plt

feature_names = up_X.columns
sorted_idx = ensemble_feature_importance.argsort()[::-1]
top_n = 10
top_feature_names = [feature_names[i] for i in sorted_idx[:top_n]]
top_feature_importance = ensemble_feature_importance[sorted_idx[:top_n]]

# bar 그래프 생성
plt.figure(figsize=(10, 6))
plt.barh(top_feature_names, top_feature_importance)
plt.xlabel('특성 중요도')
plt.ylabel('특성')
plt.title('앙상블 모델의 특성 중요도')
plt.gca().invert_yaxis()  # y축 순서 뒤집기 (중요도가 높은 특성이 위로 오도록)

plt.show()


# In[14]:


#shap value
import shap
explainer_dt = shap.Explainer(dt_model1)
shap_values_dt = explainer_dt.shap_values(up_X)

explainer_rf = shap.Explainer(rf_model1)
shap_values_rf = explainer_rf.shap_values(up_X)

explainer_gb = shap.Explainer(gb_model1)
shap_values_gb = explainer_gb.shap_values(up_X)

ensemble_shap_values = (shap_values_dt + shap_values_rf + shap_values_gb) / 3


# In[29]:


#저위험군 Ensemble Regressor fitting
low_X=low_df.drop(['고위험','사고율'],axis=1)
low_y=low_df['사고율']

dt_model2=DecisionTreeRegressor(random_state=42)
rf_model2=RandomForestRegressor(random_state=42)
gb_model2=GradientBoostingRegressor(random_state=42)

ensemble_model2=VotingRegressor(estimators=[('dt', dt_model2), ('rf', rf_model2), ('gb', gb_model2)])
ensemble_model2.fit(low_X, low_y)


# In[ ]:


#feature importance, shap value 도출을 위해 각 모델에 fitting
dt_model2.fit(low_X, low_y)
rf_model2.fit(low_X, low_y)
gb_model2.fit(low_X, low_y)


# In[ ]:


#feature importance
dt_feature_importance = dt_model2.feature_importances_
rf_feature_importance = rf_model2.feature_importances_
gb_feature_importance = gb_model2.feature_importances_

ensemble_feature_importance = (dt_feature_importance + rf_feature_importance + gb_feature_importance) / 3


# In[ ]:


feature_names = low_X.columns
sorted_idx = ensemble_feature_importance.argsort()[::-1]
top_n = 10
top_feature_names = [feature_names[i] for i in sorted_idx[:top_n]]
top_feature_importance = ensemble_feature_importance[sorted_idx[:top_n]]

plt.figure(figsize=(10, 6))
plt.barh(top_feature_names, top_feature_importance)
plt.xlabel('특성 중요도')
plt.ylabel('특성')
plt.title('앙상블 모델의 특성 중요도')
plt.gca().invert_yaxis()

plt.show()


# In[26]:


#test data에 적용
#고위험군/저위험군 모델링을 위한 데이터셋 분리
test_df=pd.read_csv('C:/Users/EunSeon/Downloads/test_dataset.csv')
up_test_df=test_df[test_df['고위험군 예측']==1]
low_test_df=test_df[test_df['고위험군 예측']==0]


# In[27]:


test_df.info()


# In[28]:


#예상 고위험군 고객 Ensemble Regressor 예측
up_X_test=up_test_df.drop(['고위험군 예측','사고율'],axis=1)
up_y_test=up_test_df['사고율']

y_pred = ensemble_model1.predict(up_X_test)

up_test_pred=pd.DataFrame(y_pred,columns=['예상 사고율'])
idx=up_X_test.index
up_test_pred.set_index(idx,inplace=True)


# In[ ]:


#예상 저위험군 고객 Ensemble Regressor 예측
low_X_test=low_test_df.drop(['고위험군 예측','사고율'],axis=1)
low_y_test=low_test_df['사고율']

y_pred = ensemble_model2.predict(low_X_test)

low_test_pred=pd.DataFrame(y_pred,columns=['예상 사고율'])
idx=low_X_test.index
low_test_pred.set_index(idx,inplace=True)


# In[ ]:


pred_res=pd.concat([up_test_pred,low_test_pred])
result=pd.concat([test_df,pred_res],axis=1)
result=result.reset_index(drop=True)


# In[ ]:


#rmse 계산
import math
rmse = math.sqrt(mean_squared_error(result['사고율'], result['예상 사고율']))
rmse


# In[ ]:


#고위험군 shap value 계산
explainer_dt = shap.Explainer(dt_model1)
shap_values_dt = explainer_dt.shap_values(up_X_test)

explainer_rf = shap.Explainer(rf_model1)
shap_values_rf = explainer_rf.shap_values(up_X_test)

explainer_gb = shap.Explainer(gb_model1)
shap_values_gb = explainer_gb.shap_values(up_X_test)

ensemble_shap_values = (shap_values_dt + shap_values_rf + shap_values_gb) / 3


# In[ ]:


shap.summary_plot(ensemble_shap_values, up_X)


# In[ ]:


up_shap_df = pd.DataFrame(ensemble_shap_values, columns=up_X_test.columns)
idx=up_X_test.index
up_shap_df.set_index(idx,inplace=True)


# In[ ]:


#저위험군 shap value 계산
explainer_dt = shap.Explainer(dt_model2)
shap_values_dt = explainer_dt.shap_values(low_X_test)

explainer_rf = shap.Explainer(rf_model2)
shap_values_rf = explainer_rf.shap_values(low_X_test)

explainer_gb = shap.Explainer(gb_model2)
shap_values_gb = explainer_gb.shap_values(low_X_test)

# 개별 모델의 Shapley 값을 통합
ensemble_shap_values = (shap_values_dt + shap_values_rf + shap_values_gb) / 3


# In[ ]:


shap.summary_plot(ensemble_shap_values, low_X)


# In[ ]:


low_shap_df = pd.DataFrame(ensemble_shap_values, columns=low_X_test.columns)
idx=low_X_test.index
low_shap_df.set_index(idx,inplace=True)


# In[ ]:


shap=pd.concat([up_shap_df,low_shap_df],axis=1)

