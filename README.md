# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dt=pd.read_csv("/content/titanic_dataset.csv")
dt
```
![Screenshot 2024-03-09 111446](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/83f4ff2c-edd2-4191-9421-bf727013ebf8)


```
dt.info()
```
![Screenshot 2024-03-09 111848](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/d75d3d97-c83d-4d2b-bbf7-071485cb2187)


```
dt.shape
```
![Screenshot 2024-03-09 111927](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/10f26b11-c859-4ca8-860d-765c5edf09de)


```
dt.set_index("PassengerId",inplace=True)
dt.describe()
```
![Screenshot 2024-03-09 112037](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/716f82d5-d97a-4666-a113-ab39c226ee2a)


```
dt.nunique()
```
![Screenshot 2024-03-09 112112](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/99249907-8415-4dee-962f-f41d75c5158c)


```
dt["Survived"].value_counts()
```
![Screenshot 2024-03-09 112211](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/646f84ff-7fb7-43d6-87df-8bf3f4e169d1)


```
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per
```
![Screenshot 2024-03-09 112247](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/c41679de-84c8-49bf-b4ad-2ab91ec9b489)


```
sns.countplot(data=dt,x="Survived")
```
![Screenshot 2024-03-09 112334](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/c2e65eb0-78c2-4cab-96f9-19b940c74ce8)


```
dt
```
![Screenshot 2024-03-09 113810](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/6970fea4-0ff2-4194-bf19-55eb3c97830a)



```
dt.Pclass.unique()
```
![Screenshot 2024-03-09 112614](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/7eb9f3c4-151b-4973-a69d-531caccd63c1)


```
dt.rename(columns={'Sex':'Gender'},inplace=True)
dt
```
![Screenshot 2024-03-09 112728](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/31820297-9d0c-4ad2-b426-add463c8e562)


```
sns.catplot(x="Gender",col="Survived",kind="count",data=dt,height=5,aspect=.7)
```
![Screenshot 2024-03-09 112816](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/71535f1a-31eb-4667-bcaf-cc7d1c6f3395)


```
sns.catplot(x='Survived',hue="Gender",data=dt,kind='count')
```
![Screenshot 2024-03-09 112845](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/fbf4a83a-0a70-4dc5-80cc-0fc6267a87ce)


```
dt.boxplot(column="Age",by="Survived")
```
![Screenshot 2024-03-09 112906](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/7645eb0b-dd11-4b75-bdd9-e3f422b522ef)


```
sns.scatterplot(x=dt["Age"],y=dt["Fare"])
```
![Screenshot 2024-03-09 113012](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/cdbb0911-3aad-40f5-a84d-0bd22de66b66)


```
sns.jointplot(x="Age",y="Fare",data=dt)
```
![Screenshot 2024-03-09 113049](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/338e74f4-f66c-4896-a6de-8cc0446337cf)


```
fig,ax1=plt.subplots(figsize=(8,5))
sns.boxplot(ax=ax1,x="Pclass",y="Age",hue="Gender",data=dt)
```
![Screenshot 2024-03-09 113142](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/0d2340f5-38c8-4242-b241-4e7c959f3b61)


```
sns.catplot(data=dt,col="Survived",x="Gender",hue="Pclass",kind="count")
```
![Screenshot 2024-03-09 113229](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/3cfc8650-d94d-4e80-80b7-c913b9ac4bdb)


```
corr=dt.corr()
sns.heatmap(corr,annot=True)
```
![Screenshot 2024-03-09 113316](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/15d8888c-2ec2-47ec-b8b0-322bce02dd74)


```
sns.pairplot(dt)
```
![Screenshot 2024-03-09 113415](https://github.com/gokulapriya632202/EXNO2DS/assets/119560302/ea0e6e4b-1a73-458b-8cbb-a83071374afe)

# RESULT
Thus, the Exploratory Data Analysis on the given data set was performed successfully.
