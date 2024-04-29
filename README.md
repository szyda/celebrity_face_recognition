# Celebrity face recognition
## Overview
The aim of this project is to create an application that can recognize celebrities using advanced machine learning and deep learning techniques. It automatically identifies individuals from images sourced from media, film, and television.


## Key Features
- **Architecture**: Utilizes the ResNet50 architecture pre-trained on ImageNet, which allows for high accuracy and rapid processing even with limited training data. 
- **Comprehensive Data Preprocessing**: Automated scripts for image normalization, resizing, and face detection streamline the preparation of the input data, ensuring optimal model performance.
- **Advanced Data Augmentation**: Implements image processing techniques that help to significantly enhance the model’s ability to generalize from training data to unseen real-world images. This includes rotations, shifts, flips, and more to simulate various scenarios. 
- **Scalable and flexible system**: Designed to be easily scalable to handle larger datasets and more celebrity classes. The modular design allows for easy customization and integration into existing projects. 
- **High performance**: Achieves high metrics in accuracy, precision, and recall, which are documented and verified through rigorous testing against a validation dataset. 

## Implementation
The implementation of the application revolves around several components:
- **Data preprocessing**: Starting with loading and processing images from a Kaggle dataset featuring 17 celebrities. Each image is normalized and resized to a uniform size for consistent input to the model. Advanced face detection algorithms are applied to focus the training on facial features only.
- **Model architecture**: Our system is built on the ResNet50 architecture, modified for our specific needs by removing the top layer and replacing it with layers designed to handle our 17-class problem. This includes a global average pooling layer followed by dense layers with dropout to combat overfitting.
- **Training and validation**: The model is trained using a split of 70% training and 30% validation data, allowing us to monitor and mitigate overfitting. We utilize data augmentation techniques such as rotations, shifts, and flips during training to enhance the model's ability to generalize to new images.

## Performance metrics
Our celebrity face recognition model achieved training accuracy of 93% and a validation accuracy of 90%. These metrics were visualized using TensorBoard, providing insights into the model's performance throughout the training process. 

The model was trained on a MacBook Pro 2017, including an Intel i5 processor, 8GB RAM, and Intel HD Graphics 630.

![Screenshot 2024-04-29 at 21.33.59.png](https://github.com/szyda/celebrity_face_recognition/blob/main/Screenshot%202024-04-29%20at%2021.33.59.png)]

## Installation
1. Cloning the repository. 
2. Installing required Python packages using a requirements.txt file.

    ``pip install -r requirements. txt``
3. Run tests

