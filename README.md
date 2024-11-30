This repository contains a model for classifying between healthy or COVID patients on basis of their pulmonary scans. The sick non-COVID patients (pneumonia, lung opacity) form a third class. This project deals there fore with a problem of classification with 3 classes.

<b>About the dataset.</b>

The data can be found at: https://www.kaggle.com/tawsifurrahman/covid19-radiography-database .

The features for each patient consist of two images:
* the their pulmonary scan (resolution: 300x300) and
* the mask of the region of the lungs (with variable resolution).

<b>About the problem.</b>

This project deals with a problem of classification with 3 classes. The three classes are: healthy participants, COVID patients, and the sick non-COVID patients (pneumonia, lung opacity). The data has been undersampled to produce balanced classes. The models were evaluated according to their accuracy, precision and recall of the COVID class. The models are interpreted by determining regions of interest using Grad-CAM. 

<b>Selected models.</b>

Two models have been selected. Both are based on Deep Learning (Convolutional Neural Networks).

The first model (called "CNN" in the report) contains 3 Convolutional layers, followed by 2 dense layers, trained from scratch. Its 
architecture is presented in the file <code>src/streamlit/architecture_cnn1.png</code>.

The second model (called "VGG" in the report) is an example use of Transfer Learning. It contains 5 pre-trained Convolutional layers, followed by 3 dense layers, trained from scratch. Its 
architecture is presented in the file <code>src/streamlit/architecture_vgg.png</code>.

<b>Results.</b>

The CNN model is relatively lightweight: the weights of the trained model form a file of 1.6MB (<code>models/CNN_weights.h5</code>), and these weights took 4 minutes to train on the training set of 5400 samples. Its scores are the following:
* accuracy: 0.83,
* global f1-score: 0.83,
* precision of the class COVID: 0.82,
* recall of the class COVID: 0.81 .

VGG is much more demanding in terms of resources: the weights of the trained model form a file of 56.5MB (<code>models/VGG16_weights.h5</code>), and these weights took 7 hours to train on the same dataset and on comparable hardware. Its scores are the following:
* accuracy: 0.85,
* global f1-score: 0.85,
* precision of the class COVID: 0.85,
* recall of the class COVID: 0.86.

These scores were achieed on the validation set. Prediction on test set is performed in live in Streamlit presentation, and the scores are comparable.

<b>Feedback and additional questions.</b>

The full report is located at: <code>summer24-analyse-pulmonaire-copy-AA/reports /Projet Analyse Pulmonaire - rapport final - images corrigees.pdf</code>.

A partial streamlit presentation is located at: <code>src/streamlit/presentation_5aug_fin.py</code>. This partial presentation contains only the first slides (introduction) and the last ones (definition of models, prediction on test set). For the content of slides in the middle, the reader is invited to ask the authors of the repository.

All questions about the source code should be adressed to its corresponding author Alexandre Aksenov:
* GitHub: Alexandre-aksenov
* Email: alexander1aksenov@gmail.com
* Linkedin: www.linkedin.com/in/alexandre-aksenov

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
