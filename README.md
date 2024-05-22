
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dt=pd.read_csv("/content/titanic_dataset.csv")
dt
dt.info()
sns.countplot(data=dt,x="Survived")
plt.show()
dt.rename(columns={'Sex':'Gender'},inplace=True)
dt
sns.catplot(x='Survived',hue="Gender",data=dt,kind='count')
dt.boxplot(column="Age",by="Survived")
sns.scatterplot(x=dt["Age"],y=dt["Fare"])
sns.jointplot(x="Age",y="Fare",data=dt)
corr=dt.corr()
sns.heatmap(corr,annot=True)
sns.pairplot(dt)
