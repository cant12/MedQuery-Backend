import os
import yaml
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

with open('./src/resources/config.yml', 'r') as file:
        config = yaml.safe_load(file)
persist_dir = config['vector_store']['persist_dir']

# modify model device as per requirement
embedding_model = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl', model_kwargs={'device' : 'mps'})

class VectorStoreHandler:
    def __init__(self):
         self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def load_vector_store():
        if not os.path.exists(persist_dir):
             raise Exception("Vectorstore does not exist at: " + persist_dir)
        return Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

    def index_docs(self, doc_list):
        splits = self.text_splitter.split_documents(doc_list)
        Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory=persist_dir)

    def index_web_pages(self, webpage_list):
        loader = WebBaseLoader(web_paths=webpage_list)
        doc_list = loader.load()
        self.index_docs(doc_list)