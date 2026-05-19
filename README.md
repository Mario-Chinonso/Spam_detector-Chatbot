# Spam Detector Chatbot (Machine Learning Project)
My Practical Machine Learning (ML)Project
# What it Does?
This is a simple Machine Learning spam detection chatbot built using Python.

It uses Natural Language Processing (NLP) to analyze text messages and classify them as either:
- Spam (suspicious or unwanted messages)
- Ham (normal and safe messages)

The model is trained using a Multinomial Naive Bayes classifier and CountVectorizer to convert text into numerical features.

It also gives a friendly chatbot response based on the prediction.


# What I Learned:
- Basic Machine Learning workflow
- Text preprocessing using NLP
- Feature extraction using CountVectorizer
- Training a classification model using Naive Bayes
- Building a simple interactive chatbot system

Overview

This project is a Machine Learning-based text classification system that detects whether a message is Spam or Ham (Not Spam).

It uses Natural Language Processing (NLP) techniques to process text data and a Multinomial Naive Bayes classifier to perform classification.

The system also includes a simple chatbot-style interface that responds based on predictions.

# How It Works
The dataset is loaded from a CSV file
Text data is cleaned and preprocessed
Text is converted into numerical features using CountVectorizer
A Naive Bayes model is trained on labeled data
User input is processed and classified as Spam or Ham
The system returns a response based on prediction.

# Model Details
Algorithm: Multinomial Naive Bayes
Feature Extraction: CountVectorizer (Bag of Words)
Task: Binary Text Classification
Output Classes:
spam
ham