# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl

# %%
# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['imc'] = (df['weight'] / (df['height']/100) ** 2 )
df['overweight'] = 0
df.loc[df['imc']>25, 'overweight'] = 1
df.drop(columns=['imc'], inplace=True)

# 3
df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0
df.loc[df['cholesterol'] > 1, 'cholesterol'] = 1
df.loc[df['gluc'] == 1, 'gluc'] = 0
df.loc[df['gluc'] > 1, 'gluc'] = 1

print(df.head())

#%%
# Abreviação de milhar para legenda
def thousands_formatter(x, pos):
    return f'{int(x / 1000)}K' if x >= 1000 else str(int(x))

# %%
# 4
def draw_cat_plot():
    # 5
    melt_grafico = pd.melt(df, id_vars='id', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    melt_grafico.rename(columns={
        'value': 'O valor está:'
    }, inplace=True)

    melt_grafico.replace({
        'cholesterol': 'colesterol', 
        'gluc': 'glicemia', 
        'smoke': 'tabagista', 
        'alco': 'etilista', 
        'active': 'ativo', 
        'overweight': 'sobrepeso'
    }, inplace=True)

    df_cat = melt_grafico

    # 6
    melt_merge = pd.merge(df,melt_grafico,on='id',how='inner')

    melt_merge['cardio'].replace({
        0: 'não',
        1: 'sim'
    }, inplace=True)

    melt_merge['O valor está:'].replace({
        0: 'normal',
        1: 'alterado'
    }, inplace=True)

    melt_merge.rename(columns={
        'cardio': 'FR CV',
        'variable': 'variável'
    }, inplace=True)
    
    df_cat = melt_merge
    
    # 7
    sns.set_style("darkgrid")
    sns.set_context("paper")
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    g = sns.catplot(
        data=melt_merge,
        col='FR CV',
        x='variável', hue='O valor está:',
        kind='count'
    )
    
    g.set_axis_labels("Variável", "Total")  
    g._legend.set_bbox_to_anchor((0.98, 0.85))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    plt.tight_layout()

    # Ajustar o título
    g.fig.suptitle("Determinantes para FR CV", y=1.05, fontsize=14)

    # 8
    fig = g
  
  
    # 9
    fig.savefig('catplot.png')
    return fig

#%%
# 10
def draw_heat_map():
    # 11: Clean data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975)) 
    ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(11, 9))
    
    # 15
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, fmt=".1f", 
    cbar_kws={"shrink": .5}, 
    annot=True, annot_kws={"size": 10, "color": "black"})

    # 16
    fig.savefig('heatmap.png')
    return fig

draw_cat_plot()