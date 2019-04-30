import csv
from sklearn.decomposition import *
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm

file_str = "/home/snazyman/stat_ds/data_science/hw6/winequality-red.csv"

# read in data
with open(file_str) as csvfile:
    data = []
    data_reader = csv.reader(csvfile)

    for row in data_reader:
        data.append(row)

    categories = data[0]
    data = data[1:]

    X = np.array(data)
    X = X.astype(np.float)
    
    # perform PCA on entire dataset
    # normalize data
    X_scaled = preprocessing.scale(X)
    
    # perform PCA
    pca_all = PCA(n_components = len(categories))
    pca_5 = PCA(n_components = 5)

    X_pca_all = pca_all.fit_transform(X_scaled)
    X_pca_5 = pca_5.fit_transform(X_scaled)
    
    # report top 5 components
    print(pca_5.components_)    
    
    # report explained variance
    print(pca_5.explained_variance_ratio_)

    # Consider the variables residual sugar, pH, and alcohol from the dataset
    # Compute the three corresponding principal components. Create a 3-D scatterplot
    # of this data with the principal components overlayed
    A = X_scaled[:,[3,8,10]]
    
    pca_3 = PCA(n_components = 3)
    A_pca_3 = pca_3.fit_transform(A)

    # first plot, original scaled data and principal vectors
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(A[:,0],A[:,1],A[:,2],color='red')
    for length, vector in zip(pca_3.explained_variance_, pca_3.components_):
        v = vector * np.sqrt(length)
        ax.quiver(pca_3.mean_[0],pca_3.mean_[1],pca_3.mean_[2], pca_3.mean_[0] + v[0], pca_3.mean_[1] + v[1], pca_3.mean_[2] + v[2])

    plt.title("original scaled data and principal vectors")        
    plt.show()

    # second plot, projected data onto principal vectors with the principal vectors
    fig2 = plt.figure()
    ax2 = Axes3D(fig2)
    ax2.scatter(A_pca_3[:,0],A_pca_3[:,1],A_pca_3[:,2],color='red')
    for length, vector in zip(pca_3.explained_variance_, pca_3.components_):
        v = vector * np.sqrt(length)
        ax2.quiver(pca_3.mean_[0], pca_3.mean_[1], pca_3.mean_[2], pca_3.mean_[0] + v[0], pca_3.mean_[1] + v[1], pca_3.mean_[2] + v[2])

    plt.title("projected data onto principal vectors with the principal vectors")    
    plt.show()
    
    # using entire dataset except for the variable quality, obtrain theh top three principal components and transform the data by reducing it to three dimensions using the three prinicpal components you found.

    # seperate data into quality and the rest
    B = X_scaled[:,:-1]
    B_label = X_scaled[:,11]

    # perform pca
    pca_3 = PCA(n_components = 3)    
    B_pca_3 = pca_3.fit_transform(B)

    # Fit a multiple regression model over the entire reduced dataset to predict quality.
    #B_pca_3 = sm.add_constant(B_pca_3)
    model = sm.OLS(B_label, B_pca_3).fit()

    print(model.summary())

    # the loadings of a component define a vector that is the direction in the feature space along which the data varies the most. The scores are the projected data points along this direction. So the loadings of each component is the B_pca_3.components_
    print(pca_3.components_)

    # repeat the above analysis on quality using sparse PCA
    Spca_3 = SparsePCA(n_components = 3)    
    B_Spca_3 = Spca_3.fit_transform(B)

    # Fit a multiple regression model over the entire reduced dataset to predict quality.
    #B_Spca_3 = sm.add_constant(B_Spca_3)
    model = sm.OLS(B_label, B_Spca_3).fit()

    print(model.summary())
    print(Spca_3.components_)

