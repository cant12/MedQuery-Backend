import os
import yaml
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

with open('./src/resources/config.yml', 'r') as file:
        config = yaml.safe_load(file)
openai_key = config['open_ai']['api_key']
persist_dir = config['vector_store']['persist_dir']

os.environ["OPENAI_API_KEY"] = openai_key

def main():
    loader = WebBaseLoader(
        web_paths=(
                # "https://lilianweng.github.io/posts/2023-06-23-agent/",
                "https://www.cancer.gov/types/breast/mammograms-fact-sheet#:~:text=A%20mammogram%20is%20an%20x,or%20images%2C%20of%20each%20breast.",
                ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory=persist_dir)

if __name__ == '__main__':
    main()