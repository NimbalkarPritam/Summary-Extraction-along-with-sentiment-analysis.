import pandas as pd
import numpy as np
import string
import statistics
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import re
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import resolve1
from pdfminer.high_level import extract_text
import io
from imblearn.over_sampling import SMOTE
from sklearn.metrics.pairwise import euclidean_distances
import streamlit as st
from PIL import Image

def main():
    st.markdown("<h1 style='text-align: center; color: white;'>Sentiment Analysis</h1>", unsafe_allow_html=True) 
    st.sidebar.markdown('Let’s start with Sentiment Analysis !!!') 
    image = Image.open('Sentiment_Analysis.jpg')
    st.sidebar.image(image)
    
if __name__ == "__main__":
   main()

st.sidebar.subheader("Choose Book:")
select = st.sidebar.selectbox("Book", ("","Mind Power","Thought Belief The Inner Human"))
page = st.sidebar.selectbox('Page Navigation', ["","Summary", "Sentiment analysis"])
 

def _create_frequency_table(text_string) -> dict:
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()
    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    return freqTable

def _score_sentences(sentences, freqTable) -> dict:
    sentenceValue = dict()
    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        word_count_in_sentence_except_stop_words = 0
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                word_count_in_sentence_except_stop_words += 1
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        if sentence[:10] in sentenceValue:
            sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] / word_count_in_sentence_except_stop_words
    return sentenceValue


def _find_average_score(sentenceValue) -> int:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]
    average = (sumValues / len(sentenceValue))
    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''
    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1
    return summary    

def run_summarization(text):
    # 1 Create the word frequency table
    freq_table = _create_frequency_table(text)

    # 2 Tokenize the sentences
    sentences = sent_tokenize(text)

    # 3 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(sentences, freq_table)

    # 4 Find the threshold
    threshold = _find_average_score(sentence_scores)

    # 5 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 2.3 * threshold)

    return summary  


if select=="Mind Power":
    text_str = extract_text("MindPower.pdf")

    if page=="Summary":
        if st.sidebar.button("Go", key="Go"):
            result = run_summarization(text_str)
            result = ''.join([i for i in result if not i.isdigit()])
            result = result.replace('www.positivestrategies.com','')
            result = re.sub(r"http\S+", "", result)
            result = re.sub(r"www\S+", "", result)
            result = re.sub(r"\S*@\S*\s?", "",result)
            result = re.sub('…', '', result)
            result = re.sub('[____]', '', result)
            result = re.sub(r'^https?:\/\/.*[\r\n]*', '', result, flags=re.MULTILINE)
            result = result.replace('©','')
            result = result.replace(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", '')
            result = result.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '')
            result = " ".join(result.split())
            st.write(str(result)) 

    if page=="Sentiment analysis":
        st.sidebar.subheader("Choose any option:")
        jump = st.sidebar.selectbox('Analysis', ["","Sentiment", "Graphical Representation"]) 
        if jump=="Sentiment":
            if st.sidebar.button("Go", key="Go"):
                text_df = pd.read_csv("mindpower.csv")
                X = text_df.drop(['Ssentiment_category'],axis = 1)
                y = text_df['Ssentiment_category']
                # Actual function
                X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)

                # SMOTE 
                smote_balance = SMOTE(k_neighbors = 10, sampling_strategy='not majority',random_state= 42)
                X_smote, y_smote = smote_balance.fit_resample(X,y)
                #st.write(y_smote.value_counts())
                X_train_smote,X_test_smote,y_train_smote,y_test_smote = train_test_split(X_smote,y_smote,test_size=0.2, random_state=42)    

                RF_smote = RandomForestClassifier()
                RF_smote_model = RF_smote.fit(X_train_smote,y_train_smote)
                Y_pred_RF_smote = RF_smote.predict(X_test)
                a = Y_pred_RF_smote.tolist()
                if mode(a)==0:
                    
                    st.markdown("<h1 style='text-align: center; color: #15F4F4;'>Negative Sentiment</h1>", unsafe_allow_html=True)
                    st.write("")
                    image1 = Image.open('sad.png')
                    col1, col2, col3 = st.columns([0.5, 1, 0.5])
                    col2.image(image1, use_column_width=True)
                elif mode(a)==1:
                    
                    st.markdown("<h1 style='text-align: center; color: #15F4F4;'>Neutral Sentiment</h1>", unsafe_allow_html=True)
                    st.write("")
                    image1 = Image.open('neutral.png')
                    col1, col2, col3 = st.columns([0.5, 1, 0.5])
                    col2.image(image1, use_column_width=True)
                else:
                    
                    st.markdown("<h1 style='text-align: center; color: #15F4F4;'>Positive Sentiment</h1>", unsafe_allow_html=True)
                    st.write("")
                    image1 = Image.open('happy.jpg')
                    col1, col2, col3 = st.columns([0.5, 1, 0.5])
                    col2.image(image1, use_column_width=True)

        if jump=="Graphical Representation":
            if st.sidebar.button("Go", key="Go"):
                
                text_df = pd.read_csv("mindpower.csv")
                X = text_df.drop(['Ssentiment_category'],axis = 1)
                y = text_df['Ssentiment_category']
                # Actual function
                X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)

                # SMOTE 
                smote_balance = SMOTE(k_neighbors = 10, sampling_strategy='not majority',random_state= 42)
                X_smote, y_smote = smote_balance.fit_resample(X,y)
                #st.write(y_smote.value_counts())
                X_train_smote,X_test_smote,y_train_smote,y_test_smote = train_test_split(X_smote,y_smote,test_size=0.2, random_state=42)    

                RF_smote = RandomForestClassifier()
                RF_smote_model = RF_smote.fit(X_train_smote,y_train_smote)
                Y_pred_RF_smote = RF_smote.predict(X_test)
                accuracy = accuracy_score(Y_pred_RF_smote,y_test)
                st.write("Accuracy : ","{:.2%}".format(accuracy))
                a = Y_pred_RF_smote.tolist()

                st.header("Graphs:")
                cols = st.columns(2)
                cmap = plt.cm.coolwarm
                custom_lines = [Line2D([0], [0], color=cmap(0.03), lw=4),
                        Line2D([0], [0], color=cmap(.85), lw=4),
                        Line2D([0], [0], color='green', lw=4)]
                fig, ax = plt.subplots(figsize=(8,8))        
                lines = sns.countplot(x=a , data =text_df)
                ax.legend(custom_lines, ['negative', 'neutral', 'positive'])
                plt.title("Countplot")
                cols[0].pyplot(fig)

                a =text_df['Ssentiment_category'].value_counts()[0]     #......0 =170
                b =text_df['Ssentiment_category'].value_counts()[1]     #......1 =429
                c =text_df['Ssentiment_category'].value_counts()[2]     #......2 =170

                fig1, ax1 = plt.subplots(figsize=(8, 6))
                label = ['Negative', 'Neutral', 'Positive']
                count = [a, b, c]
                colors = ['gold', 'yellowgreen', 'lightcoral']
                explode = (0, 0.1, 0)  # explode 2nd slice
                plt.pie(count, labels=label, autopct='%0.f%%', explode=explode, colors=colors,shadow=True, startangle=90)
                plt.title("Pie chart")
                cols[1].pyplot(fig1)    

if select=="Thought Belief The Inner Human":
    text_str = extract_text("Thought-Belief-The-Inner-Human.pdf")

    if page=="Summary":
        if st.sidebar.button("Go", key="Go"):
            result = run_summarization(text_str)
            result = ''.join([i for i in result if not i.isdigit()])
            result = result.replace('CHAPTER','')
            result = " ".join(result.split())
            st.write(str(result))          

    if page=="Sentiment analysis":

        st.sidebar.subheader("Choose any option:")
        jump = st.sidebar.selectbox('Analysis', ["","Sentiment", "Graphical Representation"]) 
        if jump=="Sentiment":  
            if st.sidebar.button("Go", key="Go"):
                text_df = pd.read_csv("sentimentfile.csv")
                X = text_df.drop(['Ssentiment_category'],axis = 1)
                y = text_df['Ssentiment_category']
                # Actual function
                X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)

                # SMOTE 
                smote_balance = SMOTE(k_neighbors = 10, sampling_strategy='not majority',random_state= 42)
                X_smote, y_smote = smote_balance.fit_resample(X,y)
                #st.write(y_smote.value_counts())
                X_train_smote,X_test_smote,y_train_smote,y_test_smote = train_test_split(X_smote,y_smote,test_size=0.2, random_state=42)    

                RF_smote = RandomForestClassifier()
                RF_smote_model = RF_smote.fit(X_train_smote,y_train_smote)
                Y_pred_RF_smote = RF_smote.predict(X_test)
                a = Y_pred_RF_smote.tolist()
                if mode(a)==0:
                    
                    st.markdown("<h1 style='text-align: center; color: #15F4F4;'>Negative Sentiment</h1>", unsafe_allow_html=True)
                    st.write("")
                    image1 = Image.open('sad.png')
                    col1, col2, col3 = st.columns([0.5, 1, 0.5])
                    col2.image(image1, use_column_width=True)
                elif mode(a)==1:
                    
                    st.markdown("<h1 style='text-align: center; color: #15F4F4;'>Neutral Sentiment</h1>", unsafe_allow_html=True)
                    st.write("")
                    image1 = Image.open('neutral.png')
                    col1, col2, col3 = st.columns([0.5, 1, 0.5])
                    col2.image(image1, use_column_width=True)
                else:
                    
                    st.markdown("<h1 style='text-align: center; color: #15F4F4;'>Positive Sentiment</h1>", unsafe_allow_html=True)
                    st.write("")
                    image1 = Image.open('happy.jpg')
                    col1, col2, col3 = st.columns([0.5, 1, 0.5])
                    col2.image(image1, use_column_width=True)

        if jump=="Graphical Representation":
            if st.sidebar.button("Go", key="Go"):
                
                text_df = pd.read_csv("sentimentfile.csv")
                X = text_df.drop(['Ssentiment_category'],axis = 1)
                y = text_df['Ssentiment_category']
                # Actual function
                X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)

                # SMOTE 
                smote_balance = SMOTE(k_neighbors = 10, sampling_strategy='not majority',random_state= 42)
                X_smote, y_smote = smote_balance.fit_resample(X,y)
                #st.write(y_smote.value_counts())
                X_train_smote,X_test_smote,y_train_smote,y_test_smote = train_test_split(X_smote,y_smote,test_size=0.2, random_state=42)    

                RF_smote = RandomForestClassifier()
                RF_smote_model = RF_smote.fit(X_train_smote,y_train_smote)
                Y_pred_RF_smote = RF_smote.predict(X_test)
                accuracy = accuracy_score(Y_pred_RF_smote,y_test)
                st.write("Accuracy : ","{:.2%}".format(accuracy))
                a = Y_pred_RF_smote.tolist()
                
                cols = st.columns(2)
                cmap = plt.cm.coolwarm
                custom_lines = [Line2D([0], [0], color=cmap(0.03), lw=4),
                        Line2D([0], [0], color=cmap(.85), lw=4),
                        Line2D([0], [0], color='green', lw=4)]
                fig, ax = plt.subplots(figsize=(8,9))        
                lines = sns.countplot(x=a , data =text_df)
                ax.legend(custom_lines, ['negative', 'neutral', 'positive'])
                plt.title("Countplot")
                cols[0].pyplot(fig)

                a =text_df['Ssentiment_category'].value_counts()[0]     
                b =text_df['Ssentiment_category'].value_counts()[1]     
                c =text_df['Ssentiment_category'].value_counts()[2]     

                fig1, ax1 = plt.subplots(figsize=(8, 6))
                label = ['Negative', 'Neutral', 'Positive']
                count = [a, b, c]
                colors = ['gold', 'yellowgreen', 'lightcoral']
                explode = (0, 0.1, 0)  # explode 2nd slice
                plt.pie(count, labels=label, autopct='%0.f%%', explode=explode, colors=colors,shadow=True, startangle=90)
                plt.title("Pie chart")
                cols[1].pyplot(fig1)


    
    

