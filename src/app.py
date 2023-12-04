from fastapi import FastAPI, HTTPException
from src.retriever import Retriever

app = FastAPI()
retriever = Retriever()

@app.get("/qna")
async def generate_answer(question: str):
    try:
        answer = retriever.get_response(question)
        # answer = "temp answer"
        response_body = {"answer": answer}
        return response_body
    except Exception as e:
        raise HTTPException(500, "Got the following error while attempting to generating answer: " + str(e))