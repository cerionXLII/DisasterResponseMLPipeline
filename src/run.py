import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sklearn.linear_model import SGDRegressor
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def LoadData(dataBasePath, databaseTableName):
    # load data from database
    engine = create_engine(dataBasePath)
    df = pd.read_sql(databaseTableName, engine)
    return df

dataBasePath = 'sqlite:///Data/DisasterResponse.db'
databaseTableName = 'messagesDataTable'

# load data
df = LoadData(dataBasePath, databaseTableName)


# load model
modelName = 'model.pkl'
model = joblib.load(modelName)
print(f'Model loaded: {modelName}')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    totalMessages = df.shape[0]
    totalRequests = df['request'].sum()
    totalOffers = df['offer'].sum()
    requestOfferNames = ['Total Messages', 'Requests', 'Offers']
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=requestOfferNames,
                    y=[totalMessages, totalRequests, totalOffers]
                )
            ],

            'layout': {
                'title': 'Number Request/Offer Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Kinds"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()