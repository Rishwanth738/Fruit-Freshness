**Freshness and Ripeness Detection using CNN**
This project uses a Convolutional Neural Network (CNN) to classify images of fruits into six categories based on their freshness and ripeness: fresh apples, rotten apples, fresh bananas, rotten bananas, fresh oranges, and rotten oranges.

**Project Overview**
The goal of this project is to detect whether a given fruit is fresh or rotten using image classification techniques. The dataset contains images of different fruits in their fresh and rotten states, and the CNN model is trained to predict the freshness of the fruits with the following categories:

Fresh Apples
Rotten Apples
Fresh Bananas
Rotten Bananas
Fresh Oranges
Rotten Oranges
Table of Contents
Project Overview
Directory Structure
Installation
Dataset Preparation
Model Training
Evaluation
Usage
Results
Future Work
Contributing

**Directory Structure**
The directory structure of this project is organized as follows:

├── data
│   ├── train
│   │   ├── freshapples
│   │   ├── rottenapples
│   │   ├── freshbanana
│   │   ├── rottenbanana
│   │   ├── freshoranges
│   │   └── rottenoranges
│   ├── test
│   │   ├── freshapples
│   │   ├── rottenapples
│   │   ├── freshbanana
│   │   ├── rottenbanana
│   │   ├── freshoranges
│   │   └── rottenoranges
├── models
│   └── cnn_model.h5
├── README.md
└── main.py

**Installation of Prerequisites**
1)Python 3.x
2)TensorFlow/Keras
3)NumPy

**Clone the Repository**
bash
git clone https://github.com/yourusername/fruit-freshness-detection.git
cd fruit-freshness-detection

**Install the Required Libraries**
pip install -r requirements.txt

**Dataset Preparation**
The dataset should be organized into training and testing sets. Each of these sets should contain subfolders for the fruit categories:

freshapples/
rottenapples/
freshbanana/
rottenbanana/
freshoranges/
rottenoranges/
If you don't have a validation set, you can use the validation_split parameter when training the model to split your training data into training and validation subsets.

**Model Training**
The model is a Convolutional Neural Network (CNN) built using Keras. It consists of several convolutional, max pooling, and dropout layers to prevent overfitting.

To train the model, run the following:
python main.py
You can adjust hyperparameters such as batch size, learning rate, and the number of epochs inside main.py.

**Example of Model Training Code**
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(train_data, validation_data=val_data, epochs=25)

**Evaluation**
After training, the model's performance is evaluated on the test set. If you have a validation set, the valid accuracy refers to how well the model performs on the validation set during training.

The test set results will provide insight into the model’s accuracy on unseen data.

**Model Testing**
You can evaluate your model by running:
python evaluate.py

**Usage**
To classify an image, load the trained model and pass an image through it:
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('path_to_image', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0  # Normalize the image

result = cnn.predict(test_image)
The result will provide the probability values for each category, and the predicted class can be extracted using np.argmax(result).

**Results**
The model achieves an accuracy of 97.19% on the training set and 96.16%% on the validation/test set. Here are the detailed metrics:

Training Accuracy: 97.19%
Validation Accuracy: 97.14%
Test Accuracy: 96.16%

**Future Work**
Some potential improvements for this project include:
Data Augmentation: Apply more transformations like rotations, zoom, and flips to generalize the model.
Fine-Tuning: Experiment with more layers, different architectures, or pretrained models like ResNet or MobileNet.
Object Detection: Extend the classification to include bounding box object detection.
Cross-Validation: Implement K-Fold cross-validation to ensure the model's robustness.
Contributing
Feel free to fork this repository and contribute by submitting a pull request.

**NOTE**: I have used Roboflow to export the dataset as a code since it was easier in Colab. Feel free to download the code and import it directly in Jupyter Notebook as well.
