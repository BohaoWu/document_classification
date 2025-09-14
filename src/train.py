import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

from dataset import Dataset
import config

def train():
    # get dataset
    dataset = Dataset(
        config.train_dataset_path,
        config.validation_dataset_path,
        config.industry_list_path
    )
    X_train, y_train = list(dataset.train_dataset.keys()), list(dataset.train_dataset.values())
    X_val,   y_val   = list(dataset.validation_dataset.keys()), list(dataset.validation_dataset.values())
    
    # train
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 10),
        sublinear_tf=True,
        norm="l2",
        min_df=2
    )
    pipe = Pipeline([
        ("tfidf", char_vectorizer), 
        ("clf", LinearSVC(C=1.0))
    ])
    pipe.fit(X_train, y_train)
    
    # eval
    pred = pipe.predict(X_val)
    print(classification_report(y_val, pred, digits=4))
    # print(confusion_matrix(y_val, pred))

def main():
    train()

if __name__ == "__main__":
    main()