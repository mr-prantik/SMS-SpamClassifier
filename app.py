# Importing necessary libraries
import streamlit as st
import pickle
import string
import nltk

nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()

from nltk.corpus import stopwords
nltk.download('stopwords')

# Loading the pickle files for model and vectorization
tfidf = pickle.load(open('./vectorizer.pkl', 'rb'))
model = pickle.load(open('./model.pkl', 'rb'))

# Title of the application
st.title("Email/SMS Spam Classifier")

# Input text box
input_sms = st.text_area("Enter the message: ")

# Function to transform input sms
def transform(text):
    # convert to lower case
    text = text.lower()

    # tokenization
    text = nltk.word_tokenize(text)

    # removing special characters
    temp = []
    for t in text:
        if t.isalnum():
            temp.append(t)

    text = temp.copy()
    temp.clear()

    # removing stop words and special characters
    for t in text:
        if t not in stopwords.words('english') and t not in string.punctuation:
            temp.append(t)

    text = temp.copy()
    temp.clear()

    # stemming
    for t in text:
        temp.append(ps.stem(t))

    return " ".join(temp)

if(st.button('Classify')):
    # 1. preprocessing
    transformed_sms= transform(input_sms)

    # 2. vectorization
    vector_input= tfidf.transform([transformed_sms])

    # 3. prediction
    result= model.predict(vector_input)

    # 4. display
    if result==1:
        st.header(":red[Spam]")
    else:
        st.header(":green[Not Spam]")
