import numpy as nppip
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
import joblib

import nltk
import re

from nltk.corpus import stopwords # 
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

print(stop_words)

df = pd.read_csv('IMDB Dataset.csv')

df.shape
df.info()
df["review"].value_counts()
df["sentiment"].value_counts()

# sentiment to numerical value
df["sentiment"] = df["sentiment"].map({
    "positive" : 1,
    "negative" : 0
}
)
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# apply the clean text function on Review
df["cleaned_review"] = df["review"].apply(clean_text)

df["cleaned_review"] 
# feature extraction
vectorizer = CountVectorizer(max_features = 5000)
X = vectorizer.fit_transform(df["cleaned_review"])
y = df["sentiment"]

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

# Train the model
model = MultinomialNB()
model.fit(x_train,y_train)

# prediction
y_pred = model.predict(x_test)

# calculate the performance metrics
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred) 

# print all performance
print("The Accuracy is :",accuracy)
print("The Precision is :",precision)
print("The Recall is:",recall)
print("F1 Score:",f1)
print("Confusion Matrix:\n",cm)
print("Classification Report:\n",cr)

joblib.dump(model,"sentiment_model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")
print("model has been saved")