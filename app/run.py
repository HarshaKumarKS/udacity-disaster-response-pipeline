import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, word_tokenize
import nltk

from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

#tokenizer
def tokenize(text):

    '''
    INPUT 
        text: Text to process  

    OUTPUT
        Returns a tokenized, stripped, lower cased and lammatized text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    #Stip and lammantized
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load the data
engine = create_engine('sqlite:///../data/disaster_response_db.db')
df = pd.read_sql_table('df', engine)

# load the pickle file
model = joblib.load("../models/classifier.pkl")


# take user input text for model
@app.route('/')
@app.route('/index')
def index():

    # Extract the data for visualization 
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_names = df.iloc[:,4:].columns
    category_boolean = (df.iloc[:,4:] != 0).sum().values
    
    
    # creating visualization
    graphs = [
            # genre graph
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

            # category graph    
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_boolean
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                }
            }
        }
    ]
    
    #plotly graphs are encoded in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    #web page are  rendered with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page to display results and accept inputs
@app.route('/go')
def go():
    # save the user input 
    query = request.args.get('query', '') 

    # predict the  classification using pickle model
    clf_labels = model.predict([query])[0]
    clf_results = dict(zip(df.columns[4:], clf_labels))

    # Render the html file
    return render_template(
        'go.html',
        query=query,
        classification_result=clf_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()