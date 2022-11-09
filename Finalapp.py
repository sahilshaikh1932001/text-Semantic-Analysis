
import streamlit as st
import pickle
from sentence_transformers import  util

# adding title
st.title('semantic analysis')

# adding image 
st.image("Semantic Analysis.jpg",width = 800)


# Markdown
st.markdown("#### Semantic Similarity, or Semantic Textual Similarity, is a task in the area of Natural Language Processing (NLP) that scores the relationship between texts or documents using a defined metric.There have been a lot of approaches for Semantic Similarity. The most straightforward and effective method now is to use a powerful model (e.g. transformer) to encode sentences to get their embeddings and then use a similarity metric (e.g. cosine similarity) to compute their similarity score.")

    
#model = SentenceTransformer('stsb-mpnet-base-v2')

@st.cache(allow_output_mutation = True)
def load_model():
    pickle_in = open("modelnew.pkl","rb")
    model_new = pickle.load(pickle_in)
    return model_new

model_new = load_model()

sentence_1 = st.text_input(label = 'enter the 1st sentence ', label_visibility="visible")

sentence_2 = st.text_input(label = 'enter the 2nd sentence ', label_visibility="visible")



if st.button("Predict"):
    
    # encode sentences to get their embeddings
    embedding1 = model_new.encode(sentence_1, convert_to_tensor=True)
    embedding2 = model_new.encode(sentence_2, convert_to_tensor=True)

    # compute similarity scores of two embeddings
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    
    st.success('similarity score is : {}'.format(cosine_scores.item()))

