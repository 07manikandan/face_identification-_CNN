///////////////////////////////face_identity_verification usgin CNN ///////////////////////////////////////////////////////////////
This is a face identity verification system using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. It verifies whether a given input image matches any of the faces stored in a dataset. If a match is found (based on cosine similarity of feature vectors), the person's details are displayed and spoken using text-to-speech (TTS).

////////////////////////////// Features /////////////////////////////////////////////////////////
Train a CNN model to classify known faces.

Extract feature vectors from the trained model.

Calculate cosine similarity to verify identity of a new face image.

If matched (≥ 90% similarity), display and speak out person’s name, age, and gender.

Save the trained model for reuse without retraining.

Handles missing images and missing trained model files gracefully.

///////////////////// Requirements /////////////////////////////////////////////
Python 3.7+

The following libraries:

tensorflow

numpy

pandas

scikit-learn

pyttsx3

Pillow (for image loading)
