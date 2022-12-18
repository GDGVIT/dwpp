from main import *
import uvicorn
from typing import List, Optional
from fastapi import FastAPI, Body
from pydantic import BaseModel


class Picture(BaseModel):
    link: str
    add_to_weekly_list: Optional[bool] = False


class AvgVector(BaseModel):
    category: str
    vector: List[float]


app = FastAPI()


@app.get("/")
def root():
    return {"Home": "Root"}


@app.get("/test")
def testing():
    return "Running"


@app.post('/init-vector')
def initial_vector(pic: Picture):
    img_vector = vectorize_img(pic.link)
    return {"vector": img_vector}


@app.post("/update-vector")
def update_avg_vector(pic: Picture, avg_vector: AvgVector, likes: int = Body(...)):
    img_vector = vectorize_img(pic.link)
    new_avg_vector = vector_avg(avg_vector.vector, img_vector, likes)
    return {"category": avg_vector.category, "vector": new_avg_vector}
  

@app.post("/weekly-list-approval")
def weekly_list_pic_approver(avg_vector: AvgVector, pic: Picture):
    to_add = weekly_list(avg_vector.vector, pic.link)
    return {"link": pic.link,"to_add": to_add}

if __name__ == '__main__':
    uvicorn.run(app, debug = True)
