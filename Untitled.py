
# coding: utf-8

# In[9]:


from sklearn import datasets
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
boston=datasets.load_boston()
#print(boston.DESCR)
#print(boston.data)
#print(boston.target)
#CRIM(犯罪率) ZN(房屋大於25000ft比率) INDUS(住宅比率)
#CHAS(有無鄰河) NOX(空汙比率) RM(房間數) AGE(自有住宅比率)
#DIS(離市中心距離) RAD(離高速公路距離) TAX(房屋稅率)
#PTRATIO(小學老師比例) B(黑人比率) LSTAT(低收入比率) MEDV(受雇者收入)
lr=linear_model.LinearRegression()
predicted=cross_val_predict(lr, boston.data, boston.target, cv=10)
#分成十份來訓練，其中一份為測試集
import matplotlib.pyplot as plt
y = boston.target
fig,ax=plt.subplots()
ax.scatter(y,predicted)
ax.plot([y.min(),y.max()],[y.min(),y.max()],'k--',lw=4)
plt.show()

