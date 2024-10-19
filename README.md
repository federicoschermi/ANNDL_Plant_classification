# Healthy vs Unhealthy Plants Classification

This project was developed for the course of **Artificial Neural Networks and Deep Learning** for the MSc. in Computer, Mathematical, and High-Performance Computing Engineering at Politecnico di Milano, A.Y. 2023/2024.

## Overview
The goal of this project is to build a Convolutional Neural Network (CNN) model capable of distinguishing between healthy and unhealthy plants, based on images of plant leaves.

### Dataset
The dataset used for this project can be found [here](https://www.kaggle.com/datasets/federicoschermi/an2dl-plants-dataset).

The dataset consists of 5200 images, representing healthy and unhealthy plant leaves. Each image is a 96x96 RGB tensor.

**Dataset Details:**
- **Number of Images:** 5200 total images, split into 3101 healthy and 1903 unhealthy plants after outlier removal.
- **Image Size:** 96x96 pixels, with 3 channels (RGB).
- **Train/Validation Split:** 80/20, with validation used as the test set.
- **File Format:** Images in tensor form.

### Model Requirements
- **Input/Output:** The input is a 96x96 RGB image, while the output is a single value indicating whether the plant is healthy or unhealthy. The output layer uses a sigmoid activation function for binary classification.

## Model Architecture
To build the final classifier, a combination of transfer learning and custom fully connected layers was used. The model is divided into:

### Feature Extraction Layers
The feature extraction (FE) part was implemented using a transfer learning approach. We experimented with several pretrained models, including **VGG16**, **ConvNeXtXLarge**, **EfficientNetV2L**, and **NASNetLarge**. Based on performance, **ConvNeXtXLarge** was chosen for its superior validation accuracy (0.891).

### Classification Layers
The classification layers were connected to the FE network using a **Flatten** layer followed by dense layers. The final configuration was:
- **Flatten** layer.
- **Dense Layer (1024 neurons)** with ReLU activation.
- **Dense Layer (512 neurons)** with ReLU activation.
- **Dropout Layer (0.3)** to reduce overfitting.
- **Output Layer** with 1 neuron and a sigmoid activation.

### Regularization Strategies
- **Dropout Layers:** Added after Flatten and dense layers to reduce overfitting.
- **L1-L2 Regularization:** Applied with a coefficient of 10⁻³.

## Self-Supervised Learning (SimCLR)
Due to the relatively small dataset size, a **self-supervised SimCLR** learning approach was attempted using **ResNet-18** as the encoder. This approach achieved a validation accuracy of 88.70%, but ultimately the CNN with transfer learning provided better results.

## Test Time Augmentation
To improve test performance, **test-time augmentation** was applied, generating 50 augmented versions of each original image. This led to a modest accuracy improvement of about 0.6%.

## Results
The final model achieved the following results:
- **Training Accuracy:** 99.90%
- **Validation Accuracy:** 94.60%
- **Test Accuracy:** 86.70%

**Performance Metrics on Test Set:**
- **Precision:** 83.11%
- **Recall:** 81.58%
- **F1-Score:** 82.34%

## Authors
- Luigi Pagani ([@LuigiPagani](https://github.com/LuigiPagani))
- Flavia Petruso ([@fl-hi1](https://github.com/fl-hi1))
- Federico Schermi ([@federicoschermi](https://github.com/federicoschermi))

## Output
Check out the final [`report.pdf`](./report_final.pdf).

## References
1. Ting Chen et al. "A simple framework for contrastive learning of visual representations". In: International Conference on Machine Learning (2020). [Link](https://proceedings.mlr.press/v119/chen20j.html)
2. Ian Goodfellow, Yoshua Bengio, and Aaron Courville. "Deep Learning". MIT Press (2016). [Link](https://www.deeplearningbook.org/)
3. Lisha Li et al. "Hyperband: A novel bandit-based approach to hyperparameter optimization". In: The Journal of Machine Learning Research (2017). [Link](https://jmlr.org/papers/v18/16-558.html)

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
