from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# download face dataset
faces = fetch_lfw_people(min_faces_per_person=60)
n_samples, h, w = faces.images.shape 
data = faces.data

# perform RandomPCA on the first 150 components
n_components = 150
pca = PCA(n_components=n_components,svd_solver='randomized', whiten=True)
pca_faces = pca.fit(data)

# show eigenface of first 25 prinicpal components
eigenfaces = pca_faces.components_.reshape((n_components, h, w))

# used this to plot and save eigenfaces
#for i in range(1,26):
#    plt.imshow(eigenfaces[i,:,:])
#    f = plt.figure(i+1)
    #f.savefig(f"eigenface_{i}.png")
#plt.show()

# selct some images from original dataset
Im1 = faces.images[1,:,:]
Im2 = faces.images[100,:,:]
Im3 = faces.images[500,:,:]
Im4 = faces.images[1000,:,:]
Im5 = faces.images[1300,:,:]

# project them onto the first 150 principal components
pca_Im = pca_faces.transform(faces.data)
# reconstruct from lower dimensional space to original spce
pca_reconstructed = pca_faces.inverse_transform(pca_Im).reshape((n_samples,h,w))

# grab those images
pca_Im1 = pca_reconstructed[1,:,:]
pca_Im2 = pca_reconstructed[100,:,:]
pca_Im3 = pca_reconstructed[500,:,:]
pca_Im4 = pca_reconstructed[1000,:,:]
pca_Im5 = pca_reconstructed[1300,:,:]

# plot first image and reconstruction
plt.figure(1)
plt.subplot(211)
plt.imshow(Im1)
plt.subplot(212)
plt.imshow(pca_Im1)
plt.show()

# plot image and reconstruction
plt.figure(2)
plt.subplot(211)
plt.imshow(Im2)
plt.subplot(212)
plt.imshow(pca_Im2)
plt.show()

# plot image and reconstruction
plt.figure(3)
plt.subplot(211)
plt.imshow(Im3)
plt.subplot(212)
plt.imshow(pca_Im3)
plt.show()

# plot image and reconstruction
plt.figure(4)
plt.subplot(211)
plt.imshow(Im4)
plt.subplot(212)
plt.imshow(pca_Im4)
plt.show()

# plot image and reconstruction
plt.figure(5)
plt.subplot(211)
plt.imshow(Im5)
plt.subplot(212)
plt.imshow(pca_Im5)
plt.show()
