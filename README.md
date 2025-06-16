# Dual-Branch D-WNet: Joint Optical and Time-Series SAR Fusion for Remote Sensing Classification

![image](https://github.com/user-attachments/assets/a0632f11-a607-431f-9ea1-a94cfe683025)
This project presents the **D-WNet** framework, a deep learning model designed for joint classification of optical and SAR time-series imagery for remote sensing applications. D-WNet efficiently integrates multi-source data to improve classification accuracy, particularly in environments with challenging conditions such as cloud cover, fog, and complex land cover changes.

## Overview

D-WNet uses a dual-branch architecture to process optical images and time-series SAR data separately before dynamically fusing them. This structure enables the model to better capture both spatial and temporal features, overcoming the limitations of traditional remote sensing classification methods.

### Key Features
- **Dual-Branch Architecture**: Optical and SAR data are processed in separate branches before fusion.
- **Dynamic Adaptive Fusion**: A local-global attention mechanism is used for feature fusion, dynamically adjusting based on data characteristics.
- **ConvLSTM for Temporal Modeling**: Temporal dependencies in SAR time-series data are modeled using Convolutional LSTM (ConvLSTM).
- **Residual-Guided Decoding**: Improves training stability and feature propagation.
  







Usage
Prepare the optical and SAR datasets (e.g., Sentinel-2 and Sentinel-1 data).

Preprocess the data as described in the project documentation.

Train the model using the following command:


python train.py --optical_data <path_to_optical_data> --sar_data <path_to_sar_data>
Evaluate the model's performance:


python evaluate.py --model <trained_model_path> --test_data <path_to_test_data>
Datasets
The model has been tested on datasets from three distinct regions:

1、Zaling Lake-Eling Lake Basin, Tibet

2、Gansu Province, China

3、Central-western California, USA

Each dataset includes a combination of Sentinel-1 SAR time-series data and Sentinel-2 optical imagery.

Model Performance
The D-WNet model has shown significant improvements in classification accuracy, particularly in complex environments with cloud cover and diverse land cover types. The model has been evaluated with:
![image](https://github.com/user-attachments/assets/318a4ad8-fbeb-4069-a906-6089eddcee43)
Overall Classification Accuracy (OA): 86.58% (Zaling Lake)

Kappa Coefficient: 0.8292 (Zaling Lake)


