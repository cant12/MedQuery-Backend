import os
import sys
from tqdm import tqdm
import yaml
from datasets import load_dataset, concatenate_datasets, Dataset
from langchain.schema.document import Document
from vector_store_handler import VectorStoreHandler

with open('./src/resources/config.yml', 'r') as file:
        config = yaml.safe_load(file)
dataset_path = config['dataset']['path']

def normalize(dataset, target_size, output_csv_file_path=''):
    source_set = set(dataset[:]['source'])
    source_data_list = []
    for source in source_set:
        data = dataset.filter(lambda example: example['source'] == source)
        source_data_list.append((source, data))

    source_data_list = sorted(source_data_list, key=lambda x:len(x[1]))

    cum_size = 0
    data_list = []
    for i in range(0, len(source_data_list)):
        source, data = source_data_list[i]
        size = min(len(data), (target_size-cum_size)//(len(source_data_list)-i))
        data_list.append(Dataset.from_dict(data[:size]))
        cum_size += size
    
    normalized_dataset = concatenate_datasets(data_list)
    if(output_csv_file_path != ''):
        normalized_dataset.to_csv(output_csv_file_path, index=False)
    
    return normalized_dataset

def main():
    if(os.path.exists(dataset_path)):
        print("Loading dataset from " + dataset_path)
        dataset = load_dataset(dataset_path)
    else:
         print("Dataset not found in local, Downloading from epfl-llm/guidelines")
         dataset = load_dataset("epfl-llm/guidelines")
         dataset.save_to_disk(dataset_path)

    target_size = int(sys.argv[1])
    train_data = normalize(dataset['train'], target_size, 'train_data.csv')

    print("Creating vector store")
    docs = []
    vector_store_handler = VectorStoreHandler()
    for i in tqdm(range(0, len(train_data))):
        example = train_data[i]['clean_text']
        docs.append(Document(page_content=example))
        if (i+1)%10 == 0:
            vector_store_handler.index_docs(docs)
            docs = []

if __name__ == '__main__':
    main()