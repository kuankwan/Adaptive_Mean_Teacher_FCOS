import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
plt.rc('font', **{'family': 'Microsoft YaHei, SimHei'})
plt.rcParams['axes.unicode_minus'] = False
label_list = ['person','rider','car','truck','bus','train','motorcycle','bicycle']

def heatmap(data,save_path, method='pearson', camp='RdYlGn', figsize=(10, 8), ax=None):
    """
    data: 整份数据
    method：默认为 pearson 系数
    camp：默认为：RdYlGn-红黄蓝；YlGnBu-黄绿蓝；Blues/Greens 也是不错的选择
    figsize: 默认为 10，8
    """
    input = data.data.cpu().numpy()
    data = pd.DataFrame(input).corr(method=method)
    ## 消除斜对角颜色重复的色块
    #     mask = np.zeros_like(df2.corr())
    #     mask[np.tril_indices_from(mask)] = True
    plt.figure(figsize=figsize, dpi=300)
    sns.heatmap(data, xticklabels=label_list,
                yticklabels=label_list, cmap=camp, center=0, annot=True, ax=ax)
    plt.savefig(save_path,dpi=300,bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    # x = F.normalize(x, dim=1)
    # y = F.normalize(y, dim=1)
    # similarity_matrix = torch.cosine_similarity(x, y, dim=1)
    heatmap(data=torch.randn(2048).reshape(256,8))