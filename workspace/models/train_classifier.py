import re
import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(database_filepath):
    ''' Function to read data in from the sqllite database.
    INPUT - dataframe file path 
    OUTPUT - X: message text column, Y: classification labels
    '''

    # load data from database
    dbPath = 'sqlite:///' + database_filepath
    print('db path: ', dbPath)
    engine = create_engine(dbPath)
    conn = engine.connect()

    # get data from table
    df = pd.read_sql('SELECT * FROM messages', con=conn)
    # Message Column for text
    X = df['message']
    # Classification label
    Y = df.iloc[:, 4:] 
    #column_values = Y.columns.values
    return X, Y


def tokenize(text):
    ''' Function to tokenize and normalize words from the database.
    INPUT - text
    OUTPUT - lemm: lemmatuzatinon of the input text
    '''
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    words = word_tokenize(text)

    # Remove stop words
    stop = stopwords.words("english")
    words = [t for t in words if t not in stop]

    # Lemmatization
    lemm = [WordNetLemmatizer().lemmatize(w) for w in words]

    return lemm


def build_model():
    ''' Function to build the classifier.
    OUTPUT - ada_cv: GridSearchCV
    '''
    
    # Create pipeline
    
    pipeline_ada = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'))
        ))
    ])
    
    
    # didnt work well
    #pipeline = Pipeline([
    #    ('vect', CountVectorizer(tokenizer=tokenize)),
    #    ('tfidf', TfidfTransformer()),
    #    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    #])
    
    # pipeline = Pipeline([
    # ('vect', CountVectorizer(tokenizer = tokenize))
    # , ('tfidf', TfidfTransformer())
    # , ('clf', MultiOutputClassifier(AdaBoostClassifier()))])

    
    # the parameters 
    parameters_ada = {
        'clf__estimator__learning_rate': [0.1, 0.3],
        'clf__estimator__n_estimators': [100, 200]
    }
    
    # old parameters
    #parameters = {
    #    'tfidf__use_idf': (True, False),
    #    'clf__estimator__n_estimators': [50, 60, 70]
    #}

    #cv = GridSearchCV(pipeline,
    #                  param_grid=parameters)
                      #scoring='accuracy',
                      #verbose=1,
                      #n_jobs=-1)

    cv_ada = GridSearchCV(estimator=pipeline_ada, param_grid=parameters_ada, cv=3, scoring='f1_weighted', verbose=3)            
                
    return cv_ada


def evaluate_model(model, X_test, Y_test):
    ''' Function evaluate the generated model.
    INPUT - model: model of the data, X_test: test data, Y_test: test data
    OUTPUT - ada_cv: GridSearchCV
    '''
    
    # Generate predictions
    y_pred = model.predict(X_test)

    # Print out the full classification report
    i = 0
    for col in Y_test:
        print('Feature {}: {}'.format(i + 1, col))
        print(classification_report(Y_test[col], y_pred[:, i]))
        i = i + 1
    accuracy = (y_pred == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))
    

def save_model(model, model_filepath):
    ''' Function save the generated model.
    INPUT - model: model of the data, model_filepath: model data file path
    '''

    # Save model
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        
        print('X shape...', X.shape)
        print('Y shape...', Y.shape)

        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('X_train shape...', X_train.shape)
        print('Y_train shape...', Y_train.shape)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
