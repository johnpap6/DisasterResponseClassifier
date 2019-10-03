# Disaster Response Pipeline Project

This project was developed for the Udacity Data Scientist Nanodegree.

## Project Components

There are three components in this project.

1. ETL Pipeline

In a Python script, process_data.py, write a data cleaning pipeline that:

    Loads the messages and categories datasets
    Merges the two datasets
    Cleans the data
    Stores it in a SQLite database

2. ML Pipeline

In a Python script, train_classifier.py, write a machine learning pipeline that:

    Loads data from the SQLite database
    Splits the dataset into training and test sets
    Builds a text processing and machine learning pipeline
    Trains and tunes a model using GridSearchCV
    Outputs results on the test set
    Exports the final model as a pickle file

3. Flask Web App

Flask web app containing data visualizations using Plotly in the web app.
Visualization includes the distribution of message categories, the distribution of message genres, and a visualization of the most common words contained in messages.

