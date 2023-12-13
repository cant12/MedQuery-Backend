# MedQuery-Backend

## What is it?
This is a Backend App for https://github.com/kvchandu/MedQuery

## How do I set up?
Create and activate a virtual environment with python 3.11, preferably using conda. One can install conda by following the steps [here](https://developers.google.com/earth-engine/guides/python_install-conda)
```
conda create -n med-qa python=3.11
conda activate med-qa
```

Install python requirements
```
pip install -r requirements.txt
```

Paste ur openai apikey in the src/resources/config file
```
open_ai:
  api_key: your-key
```

Run the indexing script to create the vector-store for this dataset https://huggingface.co/datasets/epfl-llm/guidelines. Give the *crop_size* argument to specify the size of the dataset you want to crop. The normalization will be performed accordingly
```
python src/index_epfl.py <crop_size>
```

Start the fast-api server
```
uvicorn src.app:app --host 0.0.0.0
```

