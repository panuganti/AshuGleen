import requests
from bs4 import BeautifulSoup

import warnings

# Turn off all warnings
warnings.filterwarnings("ignore")


def get_paragraphs_from_wikipedia(url):
    # Send a GET request to the Wikipedia page
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all the paragraphs on the page
    paragraphs = soup.find_all('p')
    return paragraphs

from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Question:  how are glacier caves formed ?
# wikipedia page - https://en.wikipedia.org/wiki/Glacier_cave   
# paragraph : ‘A glacier cave is a cave formed within the ice of a glacier. Glacier caves are often called ice caves, but the latter term is properly used to describe bedrock caves that contain year-round ice’ (summary of the page). 

# Question - how much is 1 tablespoon of water ?
# wikipedia page -https://en.wikipedia.org/wiki/Tablespoon  
# paragraph is - It has multiple answers. It could like - 
# ‘In most places, except Australia, one tablespoon equals three teaspoons—and one US tablespoon is 14.8 ml (0.50 US fl oz; 0.52 imp fl oz) or 15 ml (0.51 US fl oz; 0.53 imp fl oz).’ 
# Or
#  ‘In nutrition labeling in the U.S. and the U.K., a tablespoon is defined as 15 ml (0.51 US fl oz).[7] In Australia, the definition of the tablespoon is 20 ml (0.70 imp fl oz)’ etc.

# Question - how did anne frank die 
# wikipedia page - https://en.wikipedia.org/wiki/Anne_Frank 
# Paragraph - ‘Following their arrest, the Franks were transported to concentration camps. On 1 November 1944,[2] Anne and her sister, Margot, were transferred from Auschwitz to Bergen-Belsen concentration camp, where they died (probably of typhus) a few months later. They were originally estimated by the Red Cross to have died in March, with Dutch authorities setting 31 March as the official date. Later research has suggested they died in February or early March.’

from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

# Set a similarity score threshold -- based on test data
threshold = 0.7
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Find, filter, and sort paragraphs by similarity score
def filter_and_sort_paragraphs(question, paragraphs, threshold):
    relevant_paragraphs = []
    # Load a pre-trained model for sentence embeddings
    model_name = "paraphrase-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    question_embedding = model.encode(question, convert_to_tensor=False)
    # Encode the question and paragraphs
    non_empty_paragraphs = [p.text for p in paragraphs if p.text.strip() != ""]
    paragraph_embeddings = model.encode(non_empty_paragraphs, convert_to_tensor=False)

    # Calculate cosine similarity scores using NumPy
    similarity_scores = cosine_similarity([question_embedding], paragraph_embeddings)
    
    # Filter and sort paragraphs based on similarity score
    for i, score in enumerate(similarity_scores[0]):
        relevant_paragraphs.append((paragraphs[i], score))

    # Sort relevant paragraphs by similarity score in descending order
    relevant_paragraphs.sort(key=lambda x: x[1], reverse=True)
    return relevant_paragraphs

from transformers import pipeline
def init():
    global text2text_generator, similarity_tokenizer, similarity_model, tokenizer, model

    # Load a pre-trained model and tokenizer for text similarity
    model_name = "bert-base-uncased"
    similarity_tokenizer = AutoTokenizer.from_pretrained(model_name)
    similarity_model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    text2text_generator = pipeline("text2text-generation")

# Define a function that generates an answer based on the question and URL
def generate_answer(question, url):
    paragraphs = get_paragraphs_from_wikipedia(url)
    print("")
    print("")
    print("")
    print("")
    print("")
    # Find and rank the paragraphs by similarity score
    relevant_paragraphs = filter_and_sort_paragraphs(question, paragraphs, threshold)
    responses = []
    for paragraph, score in relevant_paragraphs:
        if (score > threshold):
            response = text2text_generator(f"question: {question}? context: {paragraph.text}")
            responses.append(response[0]['generated_text'])

    return responses

init()
print(generate_answer("how are glacier caves formed", "https://en.wikipedia.org/wiki/Glacier_cave"))
print(generate_answer("how much is 1 tablespoon of water ?", "https://en.wikipedia.org/wiki/Tablespoon"))
print(generate_answer("how did anne frank die ", "https://en.wikipedia.org/wiki/Anne_Frank"))

# Question:  how are glacier caves formed ?
# wikipedia page - https://en.wikipedia.org/wiki/Glacier_cave   
# paragraph : ‘A glacier cave is a cave formed within the ice of a glacier. Glacier caves are often called ice caves, but the latter term is properly used to describe bedrock caves that contain year-round ice’ (summary of the page). 

# Question - how much is 1 tablespoon of water ?
# wikipedia page -https://en.wikipedia.org/wiki/Tablespoon  
# paragraph is - It has multiple answers. It could like - 
# ‘In most places, except Australia, one tablespoon equals three teaspoons—and one US tablespoon is 14.8 ml (0.50 US fl oz; 0.52 imp fl oz) or 15 ml (0.51 US fl oz; 0.53 imp fl oz).’ 
# Or
#  ‘In nutrition labeling in the U.S. and the U.K., a tablespoon is defined as 15 ml (0.51 US fl oz).[7] In Australia, the definition of the tablespoon is 20 ml (0.70 imp fl oz)’ etc.

# Question - how did anne frank die 
# wikipedia page - https://en.wikipedia.org/wiki/Anne_Frank 
# Paragraph - ‘Following their arrest, the Franks were transported to concentration camps. On 1 November 1944,[2] Anne and her sister, Margot, were transferred from Auschwitz to Bergen-Belsen concentration camp, where they died (probably of typhus) a few months later. They were originally estimated by the Red Cross to have died in March, with Dutch authorities setting 31 March as the official date. Later research has suggested they died in February or early March.’

# import streamlit as st

# Streamlit UI components
# st.title("Question and URL Answer Generator")

# # Input components
# question = st.text_input("Enter a question:")
# url = st.text_input("Enter a URL:")
# submit_button = st.button("Generate Answer")

# # Check if the user has entered both a question and a URL
# if submit_button and question and url:
#     # Generate the answer
#     answer = generate_answer(question, url)

#     # Display the answer
#     st.subheader("Answer:")
#     st.write(answer)
