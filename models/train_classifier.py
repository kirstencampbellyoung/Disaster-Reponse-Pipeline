import sys
import pandas
from sklearn.metrics import classification_report

from sqlalchemy import create_engine

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# import pickle
import joblib

nltk.download(['punkt', 'stopwords', 'wordnet'])
nltk.download('omw-1.4')


def load_data(database_filepath):
    """ Load data from SQL database into pandas dataframe
    :param database_filepath:
    :return X:
    :return y:
    :return Category names:
    """
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pandas.read_sql_table('df', con=engine)

    # Define feature and target variables X and Y
    X = df['message'].values
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    """ Clean, tokenize, lemmatize, remove stop words and spaces
    :param text: text to process
    :return clean_tokens:  list of clean tokens
    """

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # remove URSL
    # Names, places, things

    # iterate through each token
    clean_tokens = []
    for tok in tokens:

        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stopwords.words("english"):
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ Create model using grid search
    :return model: Grid search object with model pipeline
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight="balanced")))
    ])

    parameters = {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2],
    }

    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=3)

    return model


def evaluate_model(model, X_test, y_test, category_names):
    """ View classification report based on  predictions from test data
    :param model: Trained model
    :param X_test: test data to predict on
    :param y_test: correct labels from test data
    :param category_names: names of categories to predict
    """
    y_pred = model.best_estimator_.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    :param model: trained model
    :param model_filepath: path for model to be saved
    """
    # pickle.dump(model, open(model_filepath, 'wb'))
    joblib.dump(model, model_filepath, compress=3)


def main():
    """ Import data, create, train, evaluate and save model"""

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        #model.fit(X_train, y_train)

        print('Evaluating model...')
        # evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' 
              'as the first argument and the filepath of the pickle file to ' 
              'save the model to as the second argument. \n\nExample: python ' 
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
