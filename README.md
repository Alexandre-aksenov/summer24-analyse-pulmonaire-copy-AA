Classification of images of pulmonary scan using CNN.
==============================

This repository contains a model for classifying between healthy or COVID patients on basis of their pulmonary scans. The sick non-COVID patients (pneumonia, lung opacity) form a third class. This project deals there fore with a problem of classification with 3 classes.

The features for each patient consist of two images: the their pulmonary scan and the mask of the region of the lungs. The data can be found at: https://www.kaggle.com/tawsifurrahman/covid19-radiography-database .

A part of the final streamlit prentation is located at: <code>src/streamlit/presentation_5aug_fin.py</code> . This presentation contains the sections
"Contexte", "Données", "Répartition par type de patient", "Répartition des sources"
(at the beginning)
and "Modélisation", "Démo", "Conclusion et perspectives"
(at the end).

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
