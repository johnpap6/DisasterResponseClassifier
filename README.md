# Disaster Response Pipeline Project

Following a natural disaster disaster response organization will typically receive millions of messages and tweets either direct or via social media right at the time when the disaster reponse organizations have the least capacity to filter these messages
and decide which are the most important.

However, these messages need to be filtered and forwared to the disaster organization that is responsible for handling each particular disaster. ML models that can help us respond to future natural disasters

In this project, the data consist of relabeled tweets and text messages from real life disasters. Each message was labeled with the disaster category that it addresses. The goal is to prepare the data with an etl pipeline and then use a machine learning pipeline to build a supervised learning model that classifies the category of each message.

The data are provided by Figure Eight

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

