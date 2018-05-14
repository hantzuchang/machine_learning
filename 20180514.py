
# coding: utf-8

# In[15]:


from sklearn import datasets
import matplotlib.pyplot as plt
data= datasets.fetch_olivetti_faces()
#print(data.DESCR)
#print(data.target)
#print(data.data)
#plt.imshow(data.images[0],cmap='gray',interpolation='nearest')
#plt.show()
#把影像變成一列
targets=data.target
data=data.images.reshape(len(data.images),-1)
#訓練資料30張臉(300張圖片)，測試資料10張臉(100張圖片)
train=data[targets<30]
test=data[targets>=30]
#從10張策是臉中，亂數選5張出來，變數test的大小變成(5,4096)
n_faces=5
from sklearn.utils import check_random_state
rng=check_random_state(4)
face_ids=rng.randint(test.shape[0],size=(n_faces, ))
test=test[face_ids, :]
#把每張訓練影像和測試影像都切割成上下兩部分:X人臉上半部分，Y人臉下半部分
n_pixels=data.shape[1]
X_train=train[:, :(n_pixels+1) // 2]
y_train=train[:, n_pixels // 2:]
X_test=test[:, :(n_pixels+1) // 2]
y_test=test[:, n_pixels // 2:]
#決定預測的演算法
from sklearn.linear_model import LinearRegression
ESTIMATORS={
    "Linear Regression":LinearRegression(),
}
y_test_predict=dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train,y_train) #模型訓練
    y_test_predict[name]=estimator.predict(X_test) #模型預測

