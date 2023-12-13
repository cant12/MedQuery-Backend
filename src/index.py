import os
from tqdm import tqdm
import yaml
from datasets import load_dataset
from langchain.schema.document import Document
from vector_store_handler import VectorStoreHandler

with open("./src/resources/config.yml", "r") as file:
    config = yaml.safe_load(file)
dataset_path = config["dataset"]["path"]


def main():
    if os.path.exists(dataset_path):
        print("Loading dataset from " + dataset_path)
        dataset = load_dataset(dataset_path)
    else:
        print("Dataset not found in local, Downloading from epfl-llm/guidelines")
        dataset = load_dataset("epfl-llm/guidelines")
        dataset.save_to_disk(dataset_path)

    print("Creating vector store")
    train_data = dataset["train"]

    docs = []
    vector_store_handler = VectorStoreHandler()
    for i in tqdm(range(0, 20)):
        example = train_data[i]["clean_text"]
        docs.append(Document(page_content=example))
        if (i + 1) % 10 == 0:
            vector_store_handler.index_docs(docs)
            docs = []


if __name__ == "__main__":
    main()
