
#Converting 4 dimnesional irisdata into 3 dimensional using PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading the dataset
dataset = pd.read_csv("C:\\Users\\Admin\\Desktop\\ineuron\\Modules\\8.Visualization\\iris.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#label encoding the Target values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#Reducing the dimensions for 3D
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X = pca.fit_transform(X)


#Visualising the 3D chart
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

# Reorder the labels to have colors matching the cluster results
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Spectral, edgecolor='k')
plt.show()
