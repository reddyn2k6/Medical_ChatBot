import streamlit as st
import os
from src.helper import get_embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pinecone import Pinecone

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Medical Chatbot", page_icon="🧠")

st.title(" Medical Chatbot")

# -------------------- SESSION STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- LOAD KEYS --------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# -------------------- LOAD MODEL + RETRIEVER --------------------
embeddings = get_embedding_model()

index = pc.Index("medical-chatbot")

docsearch = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

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

# -------------------- DISPLAY CHAT --------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------- USER INPUT --------------------
if prompt_input := st.chat_input("Ask a medical question..."):
    
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(prompt_input)
            answer = response.content
            st.markdown(answer)

    # Store bot response
    st.session_state.messages.append({"role": "assistant", "content": answer})