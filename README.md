# Disaster Response Pipeline Project

### Summary:
After disasters, there are several issues that happens. Also, there are a lot of tweets that get posted. Thus, it is hard for organizations to know when a person needing help tweeting. This project will use ETL to help organizations in classifiying messages and knowing if people needs help during disasters.

### File Descriptions
- /data/process_data.py - loading, cleaning, and processing the data
- /models/train_classifier.py - loading, tokenizing, and fitting ML models

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
