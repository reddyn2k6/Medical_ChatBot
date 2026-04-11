from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings 

#Extract Or Load PDF
def load_pdf_files(directory):
    loader=DirectoryLoader(directory,loader_cls=PyPDFLoader,glob="*.pdf")
    docs=loader.load()
    return docs

#take minimal data
def minimal_documents(documents) -> List[Document]:
    minimal_docs: List[Document] = []

    for doc in documents:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )

    return minimal_docs



    #Split into smaller chunks
def break_into_chunks(documents):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )

    texts=splitter.split_documents(documents)
    return texts


#download embedding model
def get_embedding_model():
    model_name='sentence-transformers/all-MiniLM-L6-v2'
    embeddings=HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings


