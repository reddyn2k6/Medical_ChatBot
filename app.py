import streamlit as st
import os
from src.helper import get_embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Title
st.title("Medical Chatbot 🧠")

# Load API keys (from Streamlit secrets later)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Load embeddings + vector store
embeddings = get_embedding_model()

docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# Load model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN
)

model = ChatHuggingFace(llm=llm)

# Prompt
system_prompt = """
You are a medical assistant.
Answer strictly based on the context.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt
    | model
)

# User input
user_input = st.text_input("Ask a question:")

if user_input:
    with st.spinner("Thinking..."):
        response = chain.invoke(user_input)
        st.write(response.content)