import os
import shutil
import yaml
from datasets import load_dataset
from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

with open('./src/resources/config.yml', 'r') as file:
        config = yaml.safe_load(file)
openai_key = config['open_ai']['api_key']
persist_dir = config['vector_store']['persist_dir']
dataset_path = config['dataset']['path']

os.environ["OPENAI_API_KEY"] = openai_key

def main():
    if(os.path.exists(dataset_path)):
        print("Loading dataset from " + dataset_path)
        dataset = load_dataset(dataset_path)
    else:
         print("Dataset not found in local, Downloading from epfl-llm/guidelines")
         dataset = load_dataset("epfl-llm/guidelines")
         dataset.save_to_disk(dataset_path)

    print("Creating vector store")
    train_data = dataset['train']
    docs = []
    for example in train_data[:10]['clean_text']:
        docs.append(Document(page_content=example))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory=persist_dir)

if __name__ == '__main__':
    main()