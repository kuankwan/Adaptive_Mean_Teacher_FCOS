import matplotlib.pyplot as plt
import numpy as np

x_label = [0.1,0.3,0.5,0.7]
baseline = [37827,6755,263,0]
ours = [33914,10611,6082,2768]
width = 0.35
x = np.arange(len(x_label))
fig,ax = plt.subplots()
ax.bar(x-width/2,baseline,width=width,color='#FF8247',label='Baseline')
ax.bar(x+width/2, ours, width=width,color='#7ec0ee',label='Ours')
ax.set_xlabel('Test score thresh')
ax.set_ylabel('Prediction boxes')
ax.set_xticks(x)
ax.set_xticklabels(x_label)
plt.ylim(0,40000)
ax.legend()
for x1,y1 in enumerate(baseline):
    plt.text(x1-width/2,y1+200,y1,ha='center',fontsize=12)
for x2,y2 in enumerate(ours):
    plt.text(x2+width/2,y2+200,y2,ha='center',fontsize=12)
plt.savefig('./boxes_record.jpg',dpi=300,bbox_inches='tight')
plt.show()
