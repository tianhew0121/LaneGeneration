# Model Card for Highway Traffic Scene GAN

## Model Details

### Model Description

This GAN model is designed to convert abstract lane depictions into realistic highway traffic scenes. It consists of a U-Net based generator and a PatchGAN discriminator. The model aims to generate high-fidelity and plausible traffic scenes, which can be beneficial for visualization and simulation purposes.

- **Developed by:** Tingyu Zhang, Tianhe Wang
- **Model type:** Generative Adversarial Network (GAN)
- **Language(s):** Python

### Model Sources [optional]

- **Repository:** [Link to Repository]

## Uses

### Direct Use

This model is primarily intended for generating realistic traffic scenes from abstract lane depictions. Its direct use cases include traffic simulation, virtual environment creation, and visualization in urban planning and automotive industries.

### Out-of-Scope Use

The model is not designed for real-time applications due to its computational demands. It is also not suitable for generating scenes beyond the context of highway traffic, as its training was specific to this domain.

## Bias, Risks, and Limitations

This model might inherit biases present in the training data, which could lead to less accurate or unrealistic generation of scenes underrepresented in the data. The quality of output is highly dependent on the similarity of the input to the trained data.

### Recommendations

Users should be aware of the model's limitations regarding the diversity and realism of generated scenes. It is recommended to use this model in conjunction with other checks for critical applications.

## How to Get Started with the Model

Follow example jupyter notebook for information on how to train and inference.

## Training Details

### Training Data

The model was trained on TuSimple dataset, dataset of paired abstract lane depictions and corresponding realistic highway traffic scenes.

### Training Procedure

#### Preprocessing

Preprocess data with SAM. Use custom prompt to convert generated masks into gray scale image.

#### Training Hyperparameters

- **Batch Size:** Tune accordingly to your spec
- **Gen LR:** 1e-5
- **Disc LR:** 1e-5

## Computation Used

- **Hardware Type:** Apple M2 Max
- **Hours used:** 2+ hrs for training, 15+ for preprocessing

## Technical Specifications [optional]

### Model Architecture and Objective

The GAN model comprises a U-Net architecture for the generator and a PatchGAN discriminator. The model is designed to handle input images of size (3, 128, 128).

## Model Card Authors [optional]

T.Zhang, T.Wang
