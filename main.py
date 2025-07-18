from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World from Python 3.12!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
