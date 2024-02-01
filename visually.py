import matplotlib.pyplot as plt
import math
import torch
x = [1,2,3,4,5,6,7,8]
y = [46.56,47.67,48.01,48.05,46.74,45.8,47.07,45.65]

plt.plot(x,y)
#设置横纵坐标的名称以及对应字体格式
font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 16
    }
plt.xlabel('epoch',font2)
plt.ylabel('mean Average Precision(mAP)',font2)
plt.show()
plt.savefig("ins.jpg")
plt.close()