import os
import yaml
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
from src.vector_store_handler import instructor_embeddings

with open('./src/resources/config.yml', 'r') as file:
        config = yaml.safe_load(file)
openai_key = config['open_ai']['api_key']
persist_dir = config['vector_store']['persist_dir']

os.environ["OPENAI_API_KEY"] = openai_key

class Retriever:
    def __init__(self):
        if not os.path.exists(persist_dir):
             raise Exception("Vectorstore does not exist at: " + persist_dir)
        self.vectorstore = Chroma(persist_directory=persist_dir, embedding_function=instructor_embeddings)
        self.retriever = self.vectorstore.as_retriever()
        self.prompt = hub.pull("rlm/rag-prompt")
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_response(self, question):
        rag_chain = (
            {"context": self.retriever | Retriever.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain.invoke(question)