"""
Training Classifier

"""

# importing the libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import sys
import os
import re
from sqlalchemy import create_engine
import pickle
from scipy.stats import gmean
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin


def load_data_from_db(database_filepath):
    """
    Function to load Data from the DB
    
    Inputs:
       Path to SQLite destination database  -> database_filepath  

    Output:
        dataframe of features = X
        Y dataframe of labels = Y
        Categories name = category_names
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)
    
    #Dropping child_alone as it only has zeros
    df = df.drop(['child_alone'],axis=1)
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
     # Extract X and y variables for data modelling
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns 
    return X, y, category_names


def tokenize(text,url_place_holder_string="urlplaceholder"):
    """
    Tokenization function to process the text data

    """
    
    # All urls replaced with url placeholder string
    str_url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    #Extracting URLs from the given text
    url_found = re.findall(str_url, text)
    
    # Replacing the url with a url placeholder string
    for dct_url in url_found:
        text = text.replace(dct_url, url_place_holder_string)

    # Extracting the word tokens from the given text
    tokens = nltk.word_tokenize(text)
    
    #Lemmanitizer to remove derivational forms of a word
    lmtzr = nltk.WordNetLemmatizer()

    # Return the list of clean tokens
    list_clean = [lmtzr.lemmatize(w).lower().strip() for w in tokens]
    return list_clean

# Extract begining word of the sentance by creating a feature for ML classifier
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class

    """

    def starting_verb(self, text):
        list_sentance = nltk.sent_tokenize(text)
        for sentence in list_sentance:
            tag_loc = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = tag_loc[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

#Building machine  learning 
def build_pipeline():
    """
   Fuction to build the pipeline
        
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline

def multioutput_fscore(y_true,y_pred,beta=1):
    """
    MultiOutput F1 scores
        
    Inputs:
        y_true -> Labels List
        y_prod -> Predictions List
        beta -> Beta value to be used to calculate fscore metric
    
    Output:
        f1score -> Calculation geometric mean of fscore
    """
    
    #Extract  value from the y prediction dataframe
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    
     #Extract  value from the y actuals dataframe
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
    
    list_fscore = []
    for column in range(0,y_true.shape[1]):
        c_score = fbeta_score(y_true[:,column],y_pred[:,column],beta,average='weighted')
        list_fscore.append(c_score)
        
    f1_score = np.asarray(list_fscore)
    f1_score = f1_score[f1_score<1]
    
    # Calculate the mean of f1  score
    f1_score = gmean(f1_score)
    return f1_score

# ML pipeline to test the model performance
def evaluate_pipeline(pipeline, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    Inputs:
        pipeline -> Scikit ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names (multi-output)
    """
       # Get results and add them to a dataframe.
    Y_pred = pipeline.predict(X_test)
    
    multi_f1 = multioutput_fscore(Y_test,Y_pred, beta = 1)
    overall_accuracy = (Y_pred == Y_test).mean().mean()

    print('Average overall accuracy : {0:.2f}%'.format(overall_accuracy*100))
    print('F1 score :  {0:.2f}%'.format(multi_f1*100))

    # Print classification report.
    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)
    
    for column in Y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(Y_test[column],Y_pred[column]))

#Save the model as pickle file
def save_model_as_pickle(pipeline, pickle_filepath):
    """
    Save Pipeline function
    
    Inputs:
        pipeline -> GridSearchCV
        pickle_filepath -> destination path to save .pkl file

    Output:
        Pickle File    
    
    """
    pickle.dump(pipeline, open(pickle_filepath, 'wb'))

def main():
    """
    Train Classifier Main function
    
    """

    if len(sys.argv) == 3:
        database_filepath, pickle_filepath = sys.argv[1:]
        print('Loading data from {} ...'.format(database_filepath))
        X, Y, category_names = load_data_from_db(database_filepath)

        #Split the data set into train and test set in the ratio of 80% and 20%
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        #Build the pipeline model
        print('Building the pipeline ...')
        pipeline = build_pipeline()
        
        #Trining thre model
        print('Training the pipeline ...')
        pipeline.fit(X_train, Y_train)
        
        #print the evaluation model
        print('Evaluating model...')
        evaluate_pipeline(pipeline, X_test, Y_test, category_names)

        #save the model as a pickle file
        print('Saving pipeline to {} ...'.format(pickle_filepath))
        save_model_as_pickle(pipeline, pickle_filepath)

        print('Trained model saved!')

    else:
         print("Please provide the arguments correctly: \nSample Script Execution:\n\
> python train_classifier.py ../data/disaster_response_db.db classifier.pkl \n\
Arguments Description: \n\
1) Path to SQLite destination database (e.g. disaster_response_db.db)\n\
2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl")

if __name__ == '__main__':
    main()