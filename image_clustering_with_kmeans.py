# for loading/processing the images
# tensorflow 2.0
from tensorflow.keras.utils import load_img 
from tensorflow.keras.utils import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle


path = r"FILE PATH"
# change the working directory to the path where the images are located
os.chdir(path)

# this list holds all the image filename
images = []

# creates a ScandirIterator aliased as files
with os.scandir(path) as files:
  # loops through each file in the directory
    for file in files:
        if file.name.endswith('.png') or file.name.endswith('.jpeg') or file.name.endswith('.jpg'):
          # adds only the image files to the images list
            images.append(file.name)
            
print("# of images:", len(images))
            
model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
print(model)

def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    img = np.array(img) 
    reshaped_img = img.reshape(1,224,224,3)
    
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features
   
data = {}
p = r"PATH FOR SAVE FEATURE"

# lop through each image in the dataset
for image in images:
    # try to extract the features and update the dictionary
    try:
        feat = extract_features(image, model)
        data[image] = feat
    # if something fails, save the extracted features as a pickle file (optional)
    except:
        with open(p,'wb') as file:
            pickle.dump(data,file)
            
            
            
# Clustering
def clustering(n, data):
    # get a list of the filenames
    filenames = np.array(list(data.keys()))

    # get a list of just the features
    feat = np.array(list(data.values()))

    # reshape so that there are 210 samples of 4096 vectors
    feat = feat.reshape(-1,4096)

    # reduce the amount of dimensions in the feature vector
    pca = PCA(n_components=100, random_state=22)
    pca.fit(feat)
    x = pca.transform(feat)
    # cluster feature vectors
    kmeans = KMeans(n_clusters=n, random_state=22)
    kmeans.fit(x)

    groups = {}
    for file, cluster in zip(filenames,kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
    print("클러스터 개수:", len(groups.keys()))
    return groups

# function that lets you view a cluster (based on identifier)        
def view_cluster(groups, cluster):
    plt.figure(figsize = (40,40));
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 50:
        print(f"Clipping cluster size from {len(files)} to 50")
        files = files[:48]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(8,8,index+1);
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
    plt.show()
        

groups = clustering(15, data)
for i in range(len(groups.keys())):
    print("Cluster", i+1, "Images")
    view_cluster(groups, i)
