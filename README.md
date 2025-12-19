# Project_MultitaskModel

# Introduction
![alt text](./images/image-1.png)


The "Multi-task Brain Tumor Classification and Segmentation" project focuses on building an advanced Deep Learning model to improve the diagnosis of brain tumors from Magnetic Resonance Imaging (MRI).

The core of this project is the application of Multi-task Learning, which enables a single model to simultaneously perform two critical tasks:
+ Classification: Identifying the type of tumor (e.g., Glioma, Meningioma, Pituitary).
+ Segmentation: Automatically outlining the precise location, shape, and boundaries of the tumor on the MRI scan.

By forcing the model to learn both what the tumor is and where it is at the same time, we aim to create a smarter, more efficient, and more accurate diagnostic support tool. This will help reduce the workload for clinicians and assist in better treatment planning.

# Dataset Introduction
  
**1. General Information**
Overview: BRISC 2025 is a large dataset containing 6,000 brain Magnetic Resonance Imaging (MRI) scans.
Image Type: All images are T1-weighted contrast-enhanced (T1-CE MRI), a modality that clearly highlights tumor boundaries.
Planes: The data includes all three anatomical planes (axial, sagittal, and coronal), which increases the diversity and robustness of the model.
Origin: The data was aggregated from multiple public sources (like Figshare and Br35H) and then carefully annotated by certified radiologists and medical experts, particularly to add high-quality segmentation masks.

**2. Label Characteristics for Multi-Tasking**
BRISC 2025 is perfectly structured for our multi-task model, as every data sample is associated with two types of labels:
    
A. Classification Labels (For the Classification Task): Each image is categorized into one of four (4) classes:
+ Glioma
+ Meningioma
+ Pituitary
+ No Tumor

B. Segmentation Labels (For the Segmentation Task): For the images that contain a tumor, the dataset provides a corresponding binary segmentation mask:
+ Pixels belonging to the tumor region are marked (e.g., white / value 1).
+ Pixels belonging to healthy brain tissue or the background are marked (e.g., black / value 0).

# Model Unet (ResNet50 for encoder)
![alt text](./images/image-2.png)

# Workflow
1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml
10. app.py