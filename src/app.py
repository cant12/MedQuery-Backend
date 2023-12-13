from fastapi import FastAPI, HTTPException, Request
from src.retriever import Retriever
from src.vector_store_handler import VectorStoreHandler

app = FastAPI()
retriever = Retriever()
vector_store_handler = VectorStoreHandler()

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@app.get("/ask_question")
def ask_question(question: str):
    # Ask the LLM the question
    answer = retriever.get_response_no_rag(question)
    response_body = {"answer": answer}
    return response_body


@app.get("/qna")
async def generate_answer(question: str):
    try:
        answer = retriever.get_response(question)
        response_body = {"answer": answer}
        return response_body
    except Exception as e:
        raise HTTPException(
            500,
            "Got the following error while attempting to generating answer: " + str(e),
        )

@app.post("/data")
async def receive_data(request: Request):
    # Process the received data
    data = await request.json()
    messages = data["messages"]
    response = retriever.generate_answer(messages)

    return {"answer": response}

@app.post("/add/webpage")
async def add_webpage(request: Request):
    try:
        request_body = await request.json()
        vector_store_handler.index_web_pages(request_body["links"])
        return {}
    except Exception as e:
        raise HTTPException(
            500, "Got the following error while attempting to index webpage: " + str(e)
        )
