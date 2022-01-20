# Face-Clustering (done under Consulting & Analytics Club, IIT Guwahati)
Clustering set of images based on the faces recognized using the DBSCAN clustering algorithm.

Face recognition and face clustering are different. When performing face recognition we are applying *supervised learning* where we have both 
- example images of faces we want to recognize along with 
- the names that correspond to each face (i.e., the “class labels”).

But in face clustering we need to perform *unsupervised learning* — we have only the faces themselves with no names/labels. 
From there we need to identify and count the number of unique people in a dataset.

- extract and quantify the faces in a dataset
- another to cluster the faces, where each resulting cluster (ideally) represents a unique individual

The model is developed in the python language.

## Acknowledgement:
- @ageitgey for the face_recognition library (follow this [link](https://github.com/ageitgey/face_recognition) to install)
  
## Dependencies:
Install manually the following dependencies:

- face_recognition
- imutils
- sklearn.cluster
- argparse
- pickle
- openCV (cv2)
- os

Or:  
`pip install -r requirements.txt`

I tested it with the following version of python:

- 3.8

# encode_images.py
encode_faces.py script will contains code used to extract a 128-d feature vector representation for each face.

**Arguments:**
- -i --dataset : The path to the input directory of faces and images.
- -e --encodings : The path to our output serialized pickle file containing the facial encodings.
- -d --detection_method : Face detection method to be used. Can be "hog" or "cnn" (Default: cnn)

**What it does**
- create a list of all imagePaths in our dataset using the dataset path provided in our command line argument.
- we compute the 128-d face encodings for each detected face in the rgb image
- For each of the detected faces + encodings, we build a dictionary that includes:
  - The path to the input image
  - The location of the face in the image (i.e., the bounding box)
  - The 128-d encoding itself
- Can be reused. write the data list to disk as a serialized encodings.pickle file

**Usage - To run**  
`python encode_faces.py`  
or  
`python encode_faces.py --dataset datasets/default_dataset --encodings encodings/encodings.pickle --detection_method "cnn"`


# cluster_faces.py
we have quantified and encoded all faces in our dataset as 128-d vectors, the next step is to cluster them into groups.
*Our hope is that each unique individual person will have their own separate cluster*

For this task we need a clustering algorithm, many clustering algorithms such as k-means and Hierarchical 
Agglomerative Clustering, require us to specify the number of clusters we seek ahead of time.
Therefore, we need to use a density-based or graph-based clustering algorithm
*Density-based spatial clustering of applications with noise (DBSCAN)*

**Arguments:**
- -e --encodings : The path to the encodings pickle file that we generated in our previous script.
- -j --jobs : DBSCAN is multithreaded and a parameter can be passed to the constructor containing the number of parallel jobs to run. A value of -1 will use all CPUs available (default).  
- -o --output : The path to the clusters of faces.

**What it does**
- Loaded the facial encodings data from disk, Organized the data as a NumPy array, Extracted the 128-d encodings from the data , placing them in a list
- create a DBSCAN object and then fit the model on the encodings
- loop to populate all the images in the database, and check the cluster and create a directory for the cluster.
- We employ the build_montages function of imutils to generate a single image montage containing a 5×5 grid of faces

**To run**  
`python cluster_faces.py`  
or  
`python cluster_faces.py --encodings encodings/encodings.pickle --jobs -1 --output results/default_result`

# Application

This can be used to cluster out the gallery in mobile applications or any other application with large number of images which makes the operation inefficient for humans.
