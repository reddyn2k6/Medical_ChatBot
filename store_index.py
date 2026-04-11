from dotenv import load_dotenv
import os   
from src.helper import break_into_chunks,get_embedding_model,minimal_documents,load_pdf_files
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec



print("hello")    

# load env
load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

# init pinecone
pc = Pinecone(api_key=api_key)

index_name = "medical-chatbot"

# create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",        # or "gcp"
            region="us-east-1"  # choose valid region
        )
    )

print("hello")    
# connect index
index = pc.Index(index_name)

#extract data
extracted_data=load_pdf_files(r"C:\Users\14697\Desktop\Medical ChatBot\Medical_ChatBot\data")    

print("hello")    

#metadata removal
final_docs=minimal_documents(extracted_data)

print("hello")    

#chunks
chunks=break_into_chunks(final_docs)

#embeddigins
embeddings=get_embedding_model()    

print("hello")    

# create vector store
vector_store = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)

print("hello")    
