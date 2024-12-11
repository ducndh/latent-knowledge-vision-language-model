
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
 
model_list=['meta-llama/Llama-3.2-3B','Qwen/Qwen2.5-3B']
dataset_list=['imdb','amazon_polarity','dbpedia_14','ag_news','multi_nli','qnli',"gimmaru/story_cloze-2016","piqa"]
x_label=['IMDB','Amazon','DBpedia','AG News','Multi NLI','QNLI',"Story-Cloze","PIQA"]
layers=29
data_ccs=np.zeros((layers,8),dtype=int)
data_logi=np.zeros((layers,8),dtype=int)
for i in range(8):
    filename='results_'+model_list[0]+'_'+dataset_list[i]+'.csv'
    filename = filename.replace('/','_')
    d=pd.read_csv('C:/Code/CS769/team/discovering_latent_knowledge-main/model_testing/performance_5samples/'+filename)
    for j in range(layers):
        data_ccs[j,i]=int(round(np.mean(d.iloc[j,3:])*100))
        data_logi[j,i]=int(round(np.mean(d.iloc[j,2])*100))

fig, axes = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[8,8,0.2]))
fig.subplots_adjust(wspace=0.01)

data_ccs=pd.DataFrame(data_ccs)
data_ccs.columns=dataset_list

data_logi=pd.DataFrame(data_logi)
data_ccs.columns=dataset_list

hmap1 = sns.heatmap(data_logi,cmap="YlGnBu",annot=True,fmt="d",annot_kws={"weight": "bold"},square=True,ax=axes[0],cbar=False)
hmap2 = sns.heatmap(data_ccs,cmap="YlGnBu",annot=True,fmt="d",annot_kws={"weight": "bold"},square=True,ax=axes[1],cbar=False,yticklabels=False)

axes[0].set_title("Logistic Regression",size=10,fontweight='bold')
axes[1].set_title("CCS",size=10,fontweight='bold')

#for text in hmap.texts:
#    text.set_weight('bold')
hmap1.set_xticklabels(x_label, fontweight='bold')
hmap1.set_yticklabels(hmap1.get_yticklabels(),fontweight='bold')
plt.setp(hmap1.get_xticklabels(), rotation=45, ha='right')

hmap2.set_xticklabels(x_label, fontweight='bold')
hmap2.set_yticklabels(hmap2.get_yticklabels(),fontweight='bold')
plt.setp(hmap2.get_xticklabels(), rotation=45, ha='right')

fig.colorbar(axes[1].collections[0], cax=axes[2])
plt.tight_layout()
plt.show()
    