# Facerecognition_Using_Facenet
This repository discusses about face recognition using facenet technique.
FaceNet is a face recognition system that was described by Florian Schroff, et al. at Google in their 2015 paper titled *FaceNet: A Unified Embedding for Face Recognition and Clustering.*
It is a system that, given a picture of a face, will extract high-quality features from the face and predict a 128 element vector representation these features, called a face embedding.
These face embeddings were then used as the basis for training classifier systems on standard face recognition benchmark datasets, achieving then-state-of-the-art results.



The repository consists of datasets for training and validation which I have used for face recognition download the files and save the files in a folder named "5-celebrity-faces-dataset" which is the name I have used in my code you can change according to your use.
and it consists of three python files consisting of code which are used for face recognition.


The python file named "data_processing" loads the training dataset and saves a compressed file of the dataset.
and the other one namely "face_embeddings" takes in the previously saved compressed .npz file and extracts the 128 element vectors which are features on the train & val dataset and saves another compressed file which contains the face embeddings extracted from the both datasets. The face embeddings are extracted using pretrained facenet model in keras which can be found here.

https://drive.google.com/open?id=1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn

last one "face_prediction" loads the the face emebddings of the train dataset and compares this embeddings with that of the val dataset and  predicts the faces in the val dataset and displays the detected image.
