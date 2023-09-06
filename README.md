# Bharat_Intern_Data_Science
**Task 1: Stock Prediction using LSTM**
This project focuses on utilizing Long Short-Term Memory (LSTM) neural networks to predict stock prices, using the historical stock price data of Nestlé (NSRGF) obtained from Tiingo. The following key steps are involved in this project:

**Data Acquisition and Preprocessing:**

Historical stock price data is retrieved from Tiingo using an API key.
The dataset is loaded into a Pandas DataFrame, and its structure is explored using info(), head(), and describe() to gain insights.
Missing data is checked and found to be minimal.

**Data Visualization:**

The 'date' and 'close' columns are extracted for further analysis.
A time series plot is generated to visualize the variation in Nestlé's closing stock prices over time. The x-axis is configured to display years.

**Data Preprocessing:**

The 'close' prices are normalized using Min-Max scaling to ensure that the data falls within a specific range (0 to 1).
The dataset is split into training and testing sets, with 80% of the data used for training.

**Model Building:**

A Sequential LSTM model is constructed for time series forecasting.
The model architecture consists of two LSTM layers with 50 units each, followed by two Dense layers with 25 and 1 unit(s) respectively.
The model is compiled using the 'adam' optimizer and Mean Squared Error (MSE) loss function.

**Training the Model:**

The model is trained on the training dataset using batch size 1 and one epoch.

**Testing and Evaluation:**

The test data is prepared by sliding a window of 60 historical data points.
The trained model is used to make predictions on the test data.
Predictions are inverse transformed to obtain actual stock prices.
The Root Mean Squared Error (RMSE) is calculated to evaluate the model's accuracy in predicting stock prices.
The RMSE serves as the primary metric to assess the performance of the LSTM model. Lower RMSE values indicate better predictive accuracy. This project demonstrates a workflow for time series forecasting using LSTM neural networks and provides valuable insights into Nestlé's stock price trends.

**Task 2: Number Recognition using MNIST dataset:**

This project demonstrates a Python script that uses a deep learning model to predict handwritten digits. It utilizes the MNIST dataset for training and a trained model to make predictions on digit images stored in a folder on the desktop. The code first loads and preprocesses the MNIST dataset, trains a neural network model on it, and then applies this model to predict the digits in the images stored in the specified folder.

**Import necessary libraries:**

os for working with the file system.
cv2 for image processing.
numpy for numerical operations.
matplotlib for image visualization.
tensorflow for machine learning and deep learning.

**Load and preprocess the MNIST dataset:**

Load the MNIST dataset, which contains images of handwritten digits along with their labels.
Normalize the pixel values of the images to a range between 0 and 1 to improve training performance.

**Define a neural network model:**

Create a Sequential model using TensorFlow/Keras.
Flatten the 28x28 pixel images into a 1D array.
Add two dense layers with ReLU activation functions for feature extraction.
Add the output layer with 10 units and softmax activation for digit classification.
Compile the model using the Adam optimizer and sparse categorical cross-entropy loss.

**Train the model:**

Train the model using the training data (x_train and y_train) for three epochs.
The model learns to classify handwritten digits based on the training data.

**Evaluate the model:**

Evaluate the trained model's performance on the test data (x_test and y_test).
Print the loss and accuracy of the model on the test set.

**Load and predict on desktop images:**

Define the path to the folder containing the digit images on the desktop.
List all files in the folder with the ".png" extension.
Iterate through the image files:
Load each image using OpenCV and extract the first channel (assuming grayscale).
Invert the colors of the image if necessary.
Use the trained model to predict the digit in the image.
Display the image and the predicted digit using matplotlib.
Handle any exceptions or errors during processing.

This code is designed to showcase the prediction capabilities of a neural network model trained on the MNIST dataset. It demonstrates how to load, preprocess, and make predictions on digit images from a folder using the trained model.
