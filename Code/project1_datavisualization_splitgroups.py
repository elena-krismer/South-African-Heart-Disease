import pandas as pd
import seaborn as sns

filename = '../Data/SAheart.csv'
df = pd.read_csv(filename)

ax = sns.countplot(x="famhist", hue = "chd", data=df)
ax.set_xlabel('Family History of Heart Disease')
ax.legend(title = 'CHD')
sns.despine()

ax = sns.catplot(y="tobacco",  x="chd", kind="box", data=df)
ax.set_xlabels('Coronary heart disease response')

ax = sns.catplot(y="ldl",  x="chd", kind="box", data=df)
ax.set_xlabels('Coronary heart disease response')

ax = sns.catplot(y="sbp",  x="chd", kind="box", data=df)
ax.set_xlabels('Coronary heart disease response')

ax = sns.catplot(y="obesity",  x="chd", kind="box", data=df)
ax.set_xlabels('Coronary heart disease response')

ax = sns.catplot(y="alcohol",  x="chd", kind="box", data=df)
ax.set_xlabels('Coronary heart disease response')

ax = sns.catplot(y="age",  x="chd", kind="box", data=df)
ax.set_xlabels('Coronary heart disease response')

ax = sns.catplot(y="typea",  x="chd", kind="box", data=df)
ax.set_xlabels('Coronary heart disease response')

ax = sns.countplot(x="famhist", kind="box", data=df)
ax.set_xlabels('Coronary heart disease response')

ax = sns.countplot(x="adiposity", kind="box", data=df)
ax.set_xlabels('Coronary heart disease response')
