import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from firebase import Firebase
import requests
import base64

#configuration
config = {
    "apiKey": "AIzaSyDE7Nu4Bn_ISOX8y1bIyUPVMord6dgEpto",
    "authDomain": "test-406ce.firebaseapp.com",
    "projectId": "test-406ce",
    "storageBucket": "test-406ce.appspot.com",
    "messagingSenderId": "659422697557",
    "appId": "1:659422697557:web:26e9d001485dbea5829cdc",
    "measurementId": "G-HDBQ136BNP",
    "serviceAccount": "serviceAccount.json",
    "databaseURL" : "https://test-406ce-default-rtdb.firebaseio.com/"
}

#functions
def convert_test_data(img_path):
  
  import numpy as np
  import os
  import tensorflow as tf

  from keras.applications.resnet import preprocess_input
  from keras.preprocessing.image import ImageDataGenerator
  from keras.layers import Dense,GlobalAveragePooling2D
  from keras.models import Model

  from keras.layers import Dense,GlobalAveragePooling2D
  from keras.callbacks import EarlyStopping
  from tensorflow import keras

  IMG_SIZE = (224, 224)
  data = []
  labels = []


  # Download the model, valid alpha values [0.25,0.35,0.5,0.75,1]
  base_model = tf.keras.applications.ResNet50(input_shape=(224, 224 ,3), include_top=False, weights='imagenet')
  # Add average pooling to the base
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  model_frozen = Model(inputs=base_model.input,outputs=x)
  # model_frozen.save("/content/drive/MyDrive/AIClub/featurization_model.h5")
  # print("Model Saved")
  # model_frozen = keras.models.load_model('featurization_model.h5')

  img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_batch = np.expand_dims(img_array, axis=0)
  img_preprocessed = preprocess_input(img_batch)
  data.append(model_frozen.predict(img_preprocessed))
  print(img_path)
  labels.append(img_path)

  # Make sure dimensions are (num_samples, 1280)
  #data = [array[:,:600] for array in data]
  data = np.squeeze(np.array(data))
  labels = np.reshape(labels, (-1,1))
  return data, img,labels

def visualizer(_distance, _nbors, number,_img_array ):
  nbor_images = [f"assets/{i}.jpg" for i in range(number)]
  fig, axes = plt.subplots(1, len(_nbors)+1, figsize=(10, 5))

  for i in range(len(_nbors)+1):
      ax = axes[i]
      ax.set_axis_off()
      if i == 0:
          ax.imshow(_img_array)
          ax.set_title("Input")
      else:
          image_final = plt.imread(nbor_images[i-1])
          ax.imshow(image_final)
          # we get cosine distance, to convert to similarity we do 1 - cosine_distance
          ax.set_title(f"Sim: {1 - _distance[i-1]:.2f}")
  st.pyplot(fig)

def get_prediction(image_data):
  url = 'https://askai.aiclub.world/ebc3ac95-5f5c-4b83-a091-e92d8c40307e'
  r = requests.post(url, data=image_data)
  response = r.json()['predicted_label']
  print(response)
  return response





#heading
st.title("Image Similarity")

#subheader
st.subheader("Please upload an Image of a Cat or a Dog")

#file uploader
image = st.file_uploader("Please Upload an Image",accept_multiple_files=False, help="Upload an image to find the similar Images")

if image:
    im = Image.open(image)
    #displaying the image
    st.image(im)
    #saving the image for finding similarity
    im.save("assets/example.png")

    #getting the predictions
    with open("assets/example.png", "rb") as image:
      payload = base64.b64encode(image.read())
    response = get_prediction(payload)
    st.subheader("Category: {}".format(response))

    #Finding Neighbors

    #getting the image fectures and the image array
    data, img_array,_ = convert_test_data("assets/example.png")

    #importing the features csv
    data1 = pd.read_csv("feature_csv.csv")

    if response == "Dogs":
      #selecting features
      data_dog = data1[data1["label"] == "Dogs"]
      data_dog.reset_index(drop = True, inplace = True)
      features_dog = data_dog.iloc[:,0:-1]

      st.subheader("Neighbor Images for Dogs")
      no_nbors = st.slider("Choose the Number of Nighbor Images", 1, 5, 2)

      if no_nbors:

        #finding the Neighbors
        nn = NearestNeighbors(n_neighbors=no_nbors, metric = 'cosine')
        nn.fit(features_dog)

        #finding the distabce and neighbors for imported image
        distance, nbors = nn.kneighbors([data])
        distance = distance[0]
        nbors = nbors[0]

        #downloading the images from the firebase
        firebase = Firebase(config)
        #downloading an image to the firebase
        storage = firebase.storage()
        
        #downloading nbors
        for index, nbrs in enumerate(nbors):
            image_fb = "Images/Dogs/" + str(nbrs) + ".jpg"
            storage.child(image_fb).download(f"assets/{index}.jpg")

        visualizer(distance, nbors, no_nbors, img_array)
    
    else:
      #selecting features
      data_cat = data1[data1["label"] == "Cats"]
      data_cat.reset_index(drop = True, inplace = True)
      features_cat = data_cat.iloc[:,0:-1]

      st.subheader("Neighbor Images for Cats")
      no_nbors = st.slider("Choose the Number of Nighbor Images", 1, 5, 2)

      if no_nbors:

        #finding the Neighbors
        nn = NearestNeighbors(n_neighbors=no_nbors, metric = 'cosine')
        nn.fit(features_cat)

        #finding the distabce and neighbors for imported image
        distance, nbors = nn.kneighbors([data])
        distance = distance[0]
        nbors = nbors[0]

        #downloading the images from the firebase
        firebase = Firebase(config)
        #downloading an image to the firebase
        storage = firebase.storage()
        
        #downloading nbors
        for index, nbrs in enumerate(nbors):
            image_fb = "Images/Cats/" + str(nbrs) + ".jpg"
            storage.child(image_fb).download(f"assets/{index}.jpg")

        visualizer(distance, nbors, no_nbors, img_array)
