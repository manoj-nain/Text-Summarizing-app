#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:36:32 2020

@author: manoj
"""
import nltk
import streamlit as st
import pandas as pd
import numpy as np
#import nltk
#import re
import requests
from bs4 import BeautifulSoup as soup
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from nltk.tokenize import sent_tokenize
nltk.download('stopwords')
nltk.download('punkt')
st.title("Text Summarizer")
st.markdown("This web application creates \
summary of any blog from Medium.")
st.sidebar.title("Text Summarizer using NLP")
st.sidebar.markdown("This web application creates \
summary of any blog from Medium.")
st.subheader("Summary of Text From URL")
url = st.text_input("Enter URL Here","Type here")
#
#web scraping blog from url
@st.cache
def scraper(url):
    page = requests.get(url)
    bsobj = soup(page.content,'lxml')
    article_text = ""
    for para in bsobj.findAll('p'):
        article_text += " " + para.text

    article_text = article_text.strip()
    return article_text
# !wget http://nlp.stanford.edu/data/glove.6B.zip
# !unzip glove*.zip

#tokenize text
    
def tokentext(article_text):
    sentences = []
    sentences.append(sent_tokenize(article_text))
    sentences = [y for x in sentences for y in x]
    return sentences

#preprocess text

def prep(sentences):
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    return clean_sentences

stop_words = stopwords.words('english')
# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new
#remove stopwords from the sentences
 

#extract word vectors

def wordvec(clean_sentences):
    word_embeddings = {}
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    return sentence_vectors



def main():
    HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
    if st.button("Summarize"):
        if url != "Type here":
            article_text = scraper(url)
            sentences = tokentext(article_text)
            clean_sentences = prep(sentences)
            
            clean_sentences = [remove_stopwords(r.split()) for r in prep(sentences)]
            sentence_vectors =  wordvec(clean_sentences)
            sim_mat = np.zeros([len(sentences), len(sentences)])
            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    if i != j:
                        sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

            nx_graph = nx.from_numpy_array(sim_mat)
            scores = nx.pagerank(nx_graph)



            ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
            summary = ""
            for i in range(5):
                summary += ranked_sentences[i][1]
            st.write(summary)
               
            #html = displacy.render(summary,style="ent")
            #html = html.replace("\n\n","\n")
            #st.write(HTML_WRAPPER.format(html),unsafe_allow_html=True)
if __name__ == '__main__':
	main()
    
