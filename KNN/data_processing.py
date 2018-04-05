#data processing
import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use("Agg")

from scipy.io import savemat

# train = loadmat('.\Dataset\d01')

# test = loadmat('.\Dataset\d01_te')
#
# #查看有返回的类型和他的键
# # print("type of reuslt:",type(train))
# # print("keys:",train.keys())
# # print(train['data'].shape)
#
# print(test['data'].shape)
#
#
# data_df = pd.DataFrame(test['data'])
# data_df.to_csv("d01_te.csv")
# print(data_df.info())
# data_df.to_excel('d01_te.xlsx',sheet_name='d01')

# Fault 1 -Fault 9
is_Data= True

if is_Data:
    for i in range(10):
        print(i)
        data_te = loadmat('.\Dataset\d0'+ str(i) +'_te')
        data_tr = loadmat('.\Dataset\d0' + str(i) )
        data_te_df= pd.DataFrame(data_te['data'])
        data_tr_df = pd.DataFrame(data_tr['data'])
        data_te_df.to_csv('d0'+ str(i)  +'_te'+'.csv')
        data_tr_df.to_csv('d0' + str(i)  + '.csv')
else:
    print("data is ok!")

data = pd.read_csv('d01_te.csv')
# df1 = pd.DataFrame(data)
print(data.shape)
# data.apply(lambda x:(x-np.min(x)) / (np.max(x) - np.min(x)))
data = (data - data.min()) / (data.max() - data.min())

##画图


plt.figure('Sensor1')
x = range(data.shape[0])
# y = df1.loc[:,['0']]
y1 = data.loc[:,['0']]
y2 = data.loc[:,['1']]
y3 = data.loc[:,['2']]
y4 = data.loc[:,['3']]
y5 = data.loc[:,['4']]
y6 = data.loc[:,['5']]
y7 = data.loc[:,['6']]
plt.title('Fault 1')
print(y1)
lw = 1.5
# plt.plot(x,y1,color='red',linewidth=lw,label='Sensor1')
# plt.plot(x,y2,color='orangered',linewidth=lw,label='Sensor2')
plt.plot(x,y3,color='yellow',linewidth=lw,label='Sensor3')
# plt.plot(x,y4,color='green',linewidth=lw,label='Sensor4')
# plt.plot(x,y5,color='blue',linewidth=lw,label='Sensor5')
# plt.plot(x,y6,color='deepskyblue',linewidth=lw,label='Sensor6')
# plt.plot(x,y7,color='deeppink',linewidth=lw,label='Sensor7')
fname = 'Sensor3'
plt.xlabel(fname)
plt.savefig(fname+'.png')

# plt.show()










