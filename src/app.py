from fastapi import FastAPI, HTTPException, Request
from src.retriever import Retriever
from src.vector_store_handler import VectorStoreHandler

app = FastAPI()
retriever = Retriever()
vector_store_handler = VectorStoreHandler()

@app.get("/qna")
async def generate_answer(question: str):
    try:
        answer = retriever.get_response(question)
        response_body = {"answer": answer}
        return response_body
    except Exception as e:
        raise HTTPException(500, "Got the following error while attempting to generating answer: " + str(e))
    
@app.post("/add/webpage")
async def find_best_move(request: Request):
    try:
        request_body = await request.json()
        vector_store_handler.index_web_pages(request_body["links"])
        return {}
    except Exception as e:
        raise HTTPException(500, "Got the following error while attempting to index webpage: " + str(e))
