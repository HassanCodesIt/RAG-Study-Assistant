
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from RAG import extraction, vectordbadd, vectordbget, llm
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

os.makedirs("./savepdf", exist_ok=True)
os.makedirs("./vecDB1", exist_ok=True)


@app.get("/")
def home():
    return FileResponse("templates/index.html")


@app.post("/upload")
async def upload(subject: str = Form(...), file: UploadFile = File(...)):
    file_path = f"./savepdf/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = extraction(file_path)
    vectordbadd(text, subject)
    return {"message": "Uploaded & processed", "filename": file.filename, "subject": subject}


@app.get("/query")
def query_page():
    return FileResponse("templates/query.html")


@app.post("/query")
def query(user_query: str = Form(...), subject: str = Form(...)):
    chunks = vectordbget(subject, user_query)
    answer = llm(user_query, chunks)  # directly send list to llm
    return {"response": answer}
