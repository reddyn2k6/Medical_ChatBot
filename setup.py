from setuptools import find_packages, setup

setup(
    name="medical-chatbot",
    version="0.1.0",
    author="Nihal Reddy",
    author_email="nihalreddy.vanga@gmail.com",
    packages=find_packages(),
    install_requires=[
        "langchain",
"langchain-community",
"langchain-text-splitters",
"langchain-huggingface",
"langchain-pinecone",

"flask",
"python-dotenv",

"sentence-transformers",
"pypdf"

    ]
)