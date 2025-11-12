


# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import FileResponse
# from subjects import extraction, vectordbadd, vectordbget, llm

# app = FastAPI()

# @app.get("/")
# def home():
#     return FileResponse("templates/index.html")

# @app.post("/upload")
# async def upload(subject: str = Form(...), file: UploadFile = File(...)):

#     file_path = f"./savepdf/{file.filename}"

#     with open(file_path, "wb") as f:  # default code to save pdf file
#         f.write(await file.read())

    

#     text = extraction(file_path)
#     new_id = vectordbadd(text, subject, file.filename)

#     return {
#         "message": "Uploaded & processed",
#         "id": new_id,
#         "filename": file.filename
#     }

# @app.get("/query")
# def query_page():
#     return FileResponse("templates/query.html")


# @app.get("/get_pdfs/{subject}")
# def get_pdf_subject(subject):
#     pdfs=get_pdf_by_subject(subject)
#     return {"pdfs":pdfs}
   


# @app.post("/query")
# def query(user_query: str = Form(...),subject: str = Form(...),filename: str = Form(...)):
#     pdf_id=get_id_by_filename(subject,filename)
#     if pdf_id is None:
#         return {"response": "Filename not found in the database."}
    
    

#     a = vectordbget(subject, pdf_id)
#     answer = llm(user_query, a)

#     return {"response": answer}


from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from subjects import extraction, vectordbadd, vectordbget, llm
import os

app = FastAPI()

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
