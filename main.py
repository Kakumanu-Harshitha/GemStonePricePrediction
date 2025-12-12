# src/DiamondPricePrediction/app.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

# don't import heavy pipeline at module import time
# from src.DiamondPricePrediction.pipelines.prediction_pipeline import PredictPipeline, CustomData

app = FastAPI()
# templates directory (project root "templates" folder)
templates = Jinja2Templates(directory="templates")

# lazy predictor singleton
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        print("Loading predictor (this may take a few seconds)...")
        from src.DiamondPricePrediction.pipelines.prediction_pipeline import PredictPipeline
        _predictor = PredictPipeline()
        print("Predictor loaded.")
    return _predictor

# Pydantic model for JSON API
class PredictRequest(BaseModel):
    carat: float
    depth: float
    table: float
    x: float
    y: float
    z: float
    cut: str
    color: str
    clarity: str

# Home (renders index.html)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# GET form page (renders form.html)
@app.get("/predict", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# POST from HTML form (submits form fields)
@app.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    carat: float = Form(...),
    depth: float = Form(...),
    table: float = Form(...),
    x: float = Form(...),
    y: float = Form(...),
    z: float = Form(...),
    cut: str = Form(...),
    color: str = Form(...),
    clarity: str = Form(...),
):
    # build dataframe using your helper class
    from src.DiamondPricePrediction.pipelines.prediction_pipeline import CustomData

    data = CustomData(
        carat=carat,
        depth=depth,
        table=table,
        x=x,
        y=y,
        z=z,
        cut=cut,
        color=color,
        clarity=clarity,
    )
    final_data = data.get_data_as_dataframe()

    # lazy load predictor (first request may be slower)
    predictor = get_predictor()
    pred = predictor.predict(final_data)
    result = round(pred[0], 2)

    return templates.TemplateResponse(
        "result.html", {"request": request, "final_result": result}
    )

# JSON API: POST /api/predict
@app.post("/api/predict")
async def api_predict(req: PredictRequest):
    # build input DataFrame
    from src.DiamondPricePrediction.pipelines.prediction_pipeline import CustomData

    data = CustomData(
        carat=req.carat,
        depth=req.depth,
        table=req.table,
        x=req.x,
        y=req.y,
        z=req.z,
        cut=req.cut,
        color=req.color,
        clarity=req.clarity,
    )
    final_data = data.get_data_as_dataframe()

    predictor = get_predictor()
    pred = predictor.predict(final_data)

    return {"predicted_price": float(round(pred[0], 2))}
