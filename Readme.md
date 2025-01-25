# Gesture Recognition for Smart TV Control

## Overview

As part of a project aimed at revolutionizing the smart TV experience, this repository provides a gesture recognition system for controlling a smart TV using a webcam. The goal of the system is to enable users to interact with their TV seamlessly through gesture recognition, eliminating the need for a traditional remote control. This project utilizes machine learning algorithms to recognize five specific gestures, each mapped to a unique action. The gestures recognized are:

- **Thumbs Up**: Increases the volume of the TV.
- **Thumbs Down**: Decreases the volume of the TV.
- **Left Swipe**: Rewinds video content by 10 seconds.
- **Right Swipe**: Fast forwards video content by 10 seconds.
- **Stop**: Pauses the video.

The project leverages machine learning techniques to analyze the data captured by the webcam and accurately interpret user gestures, allowing for hands-free control of the TV.

## Problem Statement

In a world where interaction with electronics is increasingly becoming touch-free, there is a need for a gesture-based control system that allows smart TV users to interact without requiring a remote. A webcam mounted on the TV captures a user’s hand gestures, which are then processed in real-time to control various TV functions, such as volume and video playback.

This system is expected to:
1. Accurately detect five predefined gestures: Thumbs Up, Thumbs Down, Left Swipe, Right Swipe, and Stop.
2. Execute corresponding TV control actions like adjusting volume and rewinding/fast-forwarding videos.
3. Be highly efficient in recognizing gestures with minimal latency, ensuring smooth interaction with the TV.

## Dataset Overview

The dataset used for training this gesture recognition system consists of several hundred videos, each containing one of the five predefined gestures. Each video typically lasts 2-3 seconds and is divided into a sequence of 30 frames. These frames are used to train the machine learning model to detect spatial and temporal patterns associated with each gesture. The dataset is collected from various individuals performing the gestures, providing a diverse set of data that ensures robustness and accuracy of the model.

The dataset is pre-processed to ensure uniformity, including cropping, resizing, and normalization of the frames before feeding them into the machine learning model for training.

The dataset for the Hand Gesture Recognition System can be accessed through the following link: [Hand Gesture Recognition System Dataset](https://www.kaggle.com/code/marusagar/hand-gesture-recognition-system). It provides valuable data for training models to recognize various hand gestures, facilitating the development of gesture-based control systems.

## Technologies Used

This gesture recognition system relies on several advanced technologies that enable efficient real-time gesture detection and control. Below is a breakdown of the key technologies used:

1. **Python**: The primary programming language used for the implementation of machine learning algorithms, data preprocessing, and model training.
   
2. **TensorFlow/Keras**: TensorFlow is the deep learning framework used for model development. Keras, a high-level API for TensorFlow, was employed to design and train deep learning models. This provided an easy-to-use interface to implement complex architectures such as CNN, LSTM, GRU, and MobileNet.

3. **OpenCV**: OpenCV was used for real-time image and video processing. It enables capturing frames from the webcam and performing various image processing tasks such as resizing, cropping, and normalization, which are critical for preparing data for the model.

4. **Transfer Learning**: The project leverages transfer learning with pre-trained models such as MobileNet and GRU. This allowed us to accelerate training and improve generalization by using pre-trained weights, reducing the need for large amounts of data.

5. **Data Augmentation**: To ensure the model can generalize well, data augmentation techniques (such as random flips, rotations, and translations) were applied to the dataset. This increases the diversity of the data and helps the model better handle real-world variations in gesture inputs.

6. **NumPy and Pandas**: These libraries were utilized for efficient data manipulation, preprocessing, and handling structured data. NumPy provides optimized operations for numerical computations, while Pandas helps manage data in tabular format.

7. **Matplotlib/Seaborn**: For visualizing the training process, including loss and accuracy curves, as well as confusion matrices, these libraries were used to track the model’s performance and identify areas for improvement.

8. **Scikit-learn**: This machine learning library was used for evaluating model performance, calculating metrics like accuracy, precision, recall, and F1-score, as well as for splitting the dataset into training and validation sets.

By leveraging these technologies, the project combines machine learning, computer vision, and real-time video processing to deliver an intuitive and efficient gesture-based smart TV control system.

## Project Architecture

### 1. **Generator**
The data generator plays a critical role in preprocessing the video data and feeding it to the model. The generator is responsible for:
- **Resizing** each frame to the required input size for the model.
- **Normalization** of pixel values to a standard range (e.g., 0 to 1).
- **Augmentation** of the data to increase the robustness of the model by simulating different environmental conditions (such as lighting or angle changes).
- **Batching** the data to feed it efficiently to the model during training.

### 2. **Model Selection**
Several models were evaluated for this gesture recognition task, with a particular emphasis on balancing training accuracy, validation accuracy, and inference time. The models tested include:
- **Conv3D Architecture**: This model uses 3D convolutions, which are effective in capturing temporal patterns in video data.
- **CNN-LSTM**: A combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) units to capture both spatial and temporal features.
- **CNN-GRU with Transfer Learning**: A CNN model combined with Gated Recurrent Units (GRU) for improved learning efficiency, along with transfer learning to leverage pre-trained models.
- **MobileNet with GRU**: A lightweight MobileNet architecture used in combination with GRU layers for gesture recognition, providing efficient performance.
- **ConvLSTM**: A model that combines convolutional layers with LSTM units, making it effective for sequential data like gestures.

### 3. **Model Evaluation and Selection**
The models were evaluated based on:
- **Training Accuracy**: How well the model performed on the training set.
- **Validation Accuracy**: How well the model generalized to unseen data during validation.
- **Inference Time**: The time taken by the model to process a new gesture and make a prediction.

After testing various models, **Model 8 (Transfer Learning with MobileNet and GRU)** emerged as the top performer. This model achieved **99.92% training accuracy** and **90% validation accuracy**, demonstrating excellent performance in both training and generalization. Other models, such as **Model 4 (CNN-GRU with Transfer Learning)** and **Model 9 (TimeDistributed ConvLSTM Model)**, also showed strong results but had slightly lower validation accuracy.

## Model Summary

| **Model No.** | **Model**                             | **Training Accuracy** | **Validation Accuracy** | **Description**                                                                |
|---------------|---------------------------------------|-----------------------|-------------------------|--------------------------------------------------------------------------------|
| 1             | Conv3D Architecture                   | 61.09%                | 32%                     | Steady training improvement but stagnant validation accuracy, suggesting potential overfitting. |
| 2             | Conv3D Architecture                   | 62.59%                | 56%                     | Continuous improvement in both training and validation, with steady progress despite some fluctuation. |
| 3             | CNN-LSTM Model                        | 41.78%                | 34%                     | Steady progress in training, but validation accuracy lags, indicating a gap likely due to overfitting. |
| 4             | CNN-GRU with Transfer Learning        | 95.78%                | 80%                     | Significant improvement in both training and validation accuracy, with some gap still remaining. |
| 5             | CNN, LSTM, and GRU with Transfer Learning | 98.49%            | 75%                     | Strong training performance with consistent improvement in validation accuracy, though with some fluctuation. |
| 6             | Conv3D with Data Augmentation         | 60.86%                | 59%                     | Notable improvement in both training and validation, with consistent performance across epochs. |
| 7             | CNN, LSTM, and GRU with Transfer Learning | 98.87%             | 75%                     | Excellent training accuracy but some validation fluctuations, requiring further optimization. |
| 8             | Transfer Learning with MobileNet and GRU | 99.92%            | 90%                     | Strong training performance with fluctuating but improving validation accuracy, reflecting good generalization. |
| 9             | TimeDistributed ConvLSTM Model        | 73.15%                | 77%                     | Consistent improvement in both training and validation, suggesting effective learning and generalization. |
| 10            | TimeDistributed Conv2D with Dense     | 90.12%                | 68%                     | Strong training progress with some fluctuation in validation accuracy, indicating solid learning and generalization. |

## Top Models for Gesture Recognition

The top models for the gesture recognition task were selected based on the following criteria:
1. **Model 8 (Transfer Learning with MobileNet and GRU)**
   - **Reason**: The high training and validation accuracy make this model the best choice for real-time gesture recognition. It generalizes well and is efficient in terms of processing time.
   
2. **Model 4 (CNN-GRU with Transfer Learning)**
   - **Reason**: This model achieves strong performance with faster convergence due to transfer learning, making it a reliable option for gesture recognition with diverse input data.

3. **Model 9 (TimeDistributed ConvLSTM Model)**
   - **Reason**: The combination of convolutional and LSTM layers allows this model to effectively capture both spatial and temporal features, making it ideal for continuous gesture data.

## Conclusion

This project demonstrates the feasibility of using gesture recognition to control a smart TV. After testing various models, **Model 8 (Transfer Learning with MobileNet and GRU)** was chosen as the best performing model due to its excellent accuracy and generalization capabilities. This model, along with other strong contenders like **Model 4** and **Model 9**, shows great promise for enabling hands-free control of smart TVs.

Future improvements will focus on further optimizing the models for faster inference, handling more complex gesture sequences, and enhancing robustness across different users and environments. The ultimate goal is to provide a seamless, intuitive TV experience for users, leveraging the power of machine learning and computer vision.

## Contribution

This project is a collective effort to develop a gesture recognition system for smart TVs, aiming to provide users with a more intuitive and hands-free interaction experience. Expertise in machine learning, model optimization, and data preprocessing significantly enhanced the model’s performance. You can follow Sagar's work and contributions on his GitHub profile: [Sagar Maru](https://github.com/sagar-maru).

For the most up-to-date version of the code, please refer to the following Kaggle notebook: [Hand Gesture Recognition System Code](https://www.kaggle.com/code/marusagar/hand-gesture-recognition-system). This notebook contains the latest implementations, enhancements, and updates to the hand gesture recognition system. It provides detailed instructions and code examples to help you understand the system's functionality and how to integrate the model for gesture recognition. Feel free to explore the notebook to access the latest code and gain insights into the overall structure and performance of the system.# Hand-Gesture-Recognition
