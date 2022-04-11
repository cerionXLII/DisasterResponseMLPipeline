# DisasterResponseMLPipeline
This project demonstrates the power of Machine Learning when used as it best.
If a disaster strikes people will post Tweets or other kinds of messages online describing their needs. It could be anything from needing water, medicine or asking for other kinds of help. It would then be useful for authorities to know who is in need of help, and what kind of help they are in need of. This project is designed to automate this process. Real messages are the input to a ML-model and the output is the desired help, if any.

## ETL Pipeline
The project starts with raw text data, with corresponding labels. The data is extracted from a raw data source, cleaned and then loaded into a database.

## Machine Learning Pipeline
The cleaned data is then used as input into a machine learning pipeline. A model is created which takes the text messages as input and catagorizes it into the different kinds of needs the person may have.

## Flask App
A simple flask app is created to host the final ML-model as a web-application. The user can try it out by posting messages and the model then classifies the message into the various categories.
