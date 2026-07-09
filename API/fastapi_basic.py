from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "This is my first API"}

@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f"Hello {name}!"}

@app.get("/add/{a}/{b}")
def add(a: int, b: int):
    return {"result": a + b}