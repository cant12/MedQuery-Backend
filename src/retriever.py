import os
import yaml
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
from src.vector_store_handler import VectorStoreHandler
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage


with open("./src/resources/config.yml", "r") as file:
    config = yaml.safe_load(file)
openai_key = config["open_ai"]["api_key"]

os.environ["OPENAI_API_KEY"] = openai_key
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Retriever:
    def __init__(self):
        self.vectorstore = VectorStoreHandler.load_vector_store()
        self.retriever = self.vectorstore.as_retriever()
        self.prompt = hub.pull("rlm/rag-prompt")
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_response(self, question):
        rag_chain = (
            {
                "context": self.retriever | Retriever.format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain.invoke(question)
    
    def format_prompt(messages, question):
        template = """You are an assistant for a conversation. Use the following messages to answer the question.
            Messages:
            {messages}
            Question: {question}
            Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        message_strings = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            message_strings.append(f"{role}: {content}")
        messages_string = "\n".join(message_strings)
        return prompt.format_prompt(messages=messages_string, question=question)

    def get_response_no_rag(self, question):
        template = "You are a helpful assistant that provides Question and Answer assistance in the Medical Domain for patients."
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        msg = self.llm(
            chat_prompt.format_prompt(
                text=question,
            ).to_messages()
        )

        return msg.content

    def generate_answer_with_chat_context(self, messages):
        template = """Below is the condensed conversation between you and a patient. You are a virtual assistant for a Medical Q and A app. 
        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {
                "context": itemgetter("messages")
                | self.retriever
                | Retriever.format_docs,
                "question": itemgetter("question"),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(
            {
                "messages": str(messages[:-1]),
                "question": str(messages[-1]["content"]),
                "language": "english",
            }
        )

