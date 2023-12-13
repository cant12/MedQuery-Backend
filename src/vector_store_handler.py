import yaml
from datasets import load_dataset
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

with open('./src/resources/config.yml', 'r') as file:
        config = yaml.safe_load(file)
persist_dir = config['vector_store']['persist_dir']

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl', model_kwargs={'device' : 'mps'})

class VectorStoreHandler:
    def __init__(self):
         self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def index_docs(self, doc_list):
        splits = self.text_splitter.split_documents(doc_list)
        Chroma.from_documents(documents=splits, embedding=instructor_embeddings, persist_directory=persist_dir)

    def index_web_pages(self, webpage_list):
        loader = WebBaseLoader(web_paths=webpage_list)
        doc_list = loader.load()
        self.index_docs(doc_list)