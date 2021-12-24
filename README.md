# Project 4: Dog Breed Identification using CNNs 
Submission of Project 4 (Capstone Project) in Data Scientist Nanodegree Program - Udacity

# Installations
The code was developed using Python version 3.x. We recommend using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) for installing **Python 3** as well as other required libraries, although you can install them by other means.

Other required libraries that might not be included in the Anaconda package can be installed by:

```pip install -r requirements.txt```

If you have GPUs, please use ```requirements-gpu.txt``` instead.

Trained models can be downloaded [here](https://vnueduvn-my.sharepoint.com/:f:/g/personal/trangnm_58_vnu_edu_vn/EhzER0K8oVxAvrjDxUPsVgoBI9vkH8nlFv3NQGUHCg6abg?e=7xCzRI), put them under ```trained_models/```

If you want to train the model, download the dog dataset [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Unzip the folder and place it in the repo, at location ```data/dog_images```.

Download the human dataset [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). Unzip the folder and place it in the repo, at location ```data/lfw```

# Project Motivation
This project uses Convolutional Neural Networks (CNNs)! In this project, we will build a pipeline to process real-world, user-supplied images. Given an image of a dog, the algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

# File Descriptions

Within the download you'll find the following directories and files:
```
./
├── bottleneck_features/
│   ├── DogInceptionV3Data.npz  # bottleneck features from InceptionV3
│   └── DogVGG16Data.npz  # bottleneck features from VGG16
├── haarcascades/
│   └── haarcascade_frontalface_alt.xml  # OpenCV's pre-trained face detectors
├── images/*
├── trained_models/
│   ├── weights.best.from_scratch.hdf5  # best model trained from scratch
│   ├── weights.best.InceptionV3.hdf5  # best model transfered learning from InceptionV3
│   └── weights.best.VGG16.hdf5  # best model transfered learning from VGG16
├── dog_app.ipynb  # the main notebook containing the algorithm
├── extract_bottleneck_features.py  # utility functions for extracting bottleneck features
├── .gitignore
├── requirements.txt
├── requirements-gpu.txt
├── LICENSE.txt
└── README.md
```

# How to Use

The "dog_app.ipynb" notebook contains all the code and documentation. There are 8 main steps:

- Step 0: Import Datasets 
- Step 1: Detect Humans 
- Step 2: Detect Dogs 
- Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
- Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)
- Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)
- Step 6: Write your Algorithm 
- Step 7: Test Your Algorithm

The notebook can be used as a start for more enhancements of the application. Open the notebook:

```jupyter notebook dog_app.ipynb```

# Copyright and license
Code released under the [MIT License](https://github.com/trangnm58/ds_udacity_project_4/LICENSE.txt).