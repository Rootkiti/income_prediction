from typing import Union
from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


pickle_in = open('refined_model.pkl','rb')
model = pickle.load(pickle_in)

app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class data(BaseModel):
    locality: str
    sector: str
    occupation: str
    education_level: str
    age_category: str
    weekly_hours_category: str


@app.post("/predict")
def predict(features:data):
    y =pd.DataFrame([{
    'locality':features.locality,
    'sector':features.sector,
    'occupation':features.occupation,
    'education_level':features.education_level,
    'age_category':features.age_category,
    'weekly_hours_category':features.weekly_hours_category
    }])   

    result = model.predict(y)[0]
    return {"income_category": result}