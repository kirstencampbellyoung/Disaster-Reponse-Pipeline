# Disaster-Response-Pipeline

In this project, a machine learning algorithm was developed to classify messages that were sent during disaster events. 

This project consists of the following components:
* An ETL pipeline which processes the input datasets by merging, cleaning and loading them to a SQLite database
* An ML pipeline which loads data from the SQLite database, splits it into training and test sets, builds and trains a text processing and machine learning pipeline which is then used on a test set and exported as a pickle file,. 
* A Flask web app which visualizes the data and can be used to make predictions on new data

## Files in the respository

### data
- **disaster_categories.csv** : categories of messages data
- **disaster_messages.csv** : real disaster messages 
- **process_data.py** : ETL python script to process the data
- **DisasterResponse.db** : database in which cleaned data sits

### models

- **train_classifier.py** : NLP-ML training pipeline
- **classifier.pkl** : the final model

### app

- templates: **master.html** : main page of the web app
- templates: **go.html** : classification result page of the web app
- **run.py** : Flask file that runs the app

# How to run the app

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3000/

# Licensing & Acknowledgements

This framework was provided by Udacity as part of Data Engineering module of the Data Science Nanodegree. 