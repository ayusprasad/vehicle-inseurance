from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run
from typing import Optional
import logging

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import VehicleData, VehicleDataClassifier, predict_insurance_response
from src.pipline.training_pipeline import TrainPipeline

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class expects RAW data (not transformed) - the transformation will happen in VehicleData
    """
    def __init__(self, request: Request):
        self.request: Request = request
        self.Gender: Optional[str] = None  # Changed to string: "Male" or "Female"
        self.Age: Optional[int] = None
        self.Driving_License: Optional[int] = None
        self.Region_Code: Optional[float] = None
        self.Previously_Insured: Optional[int] = None
        self.Vehicle_Age: Optional[str] = None  # Raw: "< 1 Year", "1-2 Year", "> 2 Years"
        self.Vehicle_Damage: Optional[str] = None  # Raw: "Yes" or "No"
        self.Annual_Premium: Optional[float] = None
        self.Policy_Sales_Channel: Optional[float] = None
        self.Vintage: Optional[int] = None

    async def get_vehicle_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        """
        form = await self.request.form()
        self.Gender = form.get("Gender")
        self.Age = int(form.get("Age")) if form.get("Age") else None
        self.Driving_License = int(form.get("Driving_License")) if form.get("Driving_License") else None
        self.Region_Code = float(form.get("Region_Code")) if form.get("Region_Code") else None
        self.Previously_Insured = int(form.get("Previously_Insured")) if form.get("Previously_Insured") else None
        self.Vehicle_Age = form.get("Vehicle_Age")
        self.Vehicle_Damage = form.get("Vehicle_Damage")
        self.Annual_Premium = float(form.get("Annual_Premium")) if form.get("Annual_Premium") else None
        self.Policy_Sales_Channel = float(form.get("Policy_Sales_Channel")) if form.get("Policy_Sales_Channel") else None
        self.Vintage = int(form.get("Vintage")) if form.get("Vintage") else None

# Route to render the main page with the form
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for vehicle data input.
    """
    return templates.TemplateResponse(
        "vehicledata.html", 
        {"request": request, "context": "Rendering", "prediction": None}
    )

# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

# API endpoint for predictions (for programmatic access)
@app.post("/predict")
async def predict_api(
    Gender: str = Form(...),
    Age: int = Form(...),
    Driving_License: int = Form(...),
    Region_Code: float = Form(...),
    Previously_Insured: int = Form(...),
    Vehicle_Age: str = Form(...),
    Vehicle_Damage: str = Form(...),
    Annual_Premium: float = Form(...),
    Policy_Sales_Channel: float = Form(...),
    Vintage: int = Form(...)
):
    """
    API endpoint for vehicle insurance prediction
    """
    try:
        result = predict_insurance_response(
            Gender=Gender,
            Age=Age,
            Driving_License=Driving_License,
            Region_Code=Region_Code,
            Previously_Insured=Previously_Insured,
            Vehicle_Age=Vehicle_Age,
            Vehicle_Damage=Vehicle_Damage,
            Annual_Premium=Annual_Premium,
            Policy_Sales_Channel=Policy_Sales_Channel,
            Vintage=Vintage
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Route to handle form submission and make predictions
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """
    try:
        form = DataForm(request)
        await form.get_vehicle_data()
        
        # Validate required fields
        required_fields = [form.Gender, form.Age, form.Driving_License, form.Region_Code,
                          form.Previously_Insured, form.Vehicle_Age, form.Vehicle_Damage,
                          form.Annual_Premium, form.Policy_Sales_Channel, form.Vintage]
        
        if any(field is None for field in required_fields):
            return templates.TemplateResponse(
                "vehicledata.html",
                {
                    "request": request, 
                    "context": "Error: Please fill all fields",
                    "prediction": None
                },
            )
        
        # Make prediction using the high-level function
        result = predict_insurance_response(
            Gender=form.Gender,
            Age=form.Age,
            Driving_License=form.Driving_License,
            Region_Code=form.Region_Code,
            Previously_Insured=form.Previously_Insured,
            Vehicle_Age=form.Vehicle_Age,
            Vehicle_Damage=form.Vehicle_Damage,
            Annual_Premium=form.Annual_Premium,
            Policy_Sales_Channel=form.Policy_Sales_Channel,
            Vintage=form.Vintage
        )
        
        # Check if there was an error
        if 'error' in result:
            status = f"Error: {result['error']}"
            confidence = 0
            prediction_value = 0
        else:
            prediction_value = result['prediction']
            confidence = result['probability']
            status = "YES - Will buy insurance" if prediction_value == 1 else "NO - Will not buy insurance"
        
        # Render the same HTML page with the prediction result
        return templates.TemplateResponse(
            "vehicledata.html",
            {
                "request": request, 
                "context": status,
                "prediction": prediction_value,
                "confidence": f"{confidence:.2%}",
                "input_data": {
                    "Gender": form.Gender,
                    "Age": form.Age,
                    "Driving_License": form.Driving_License,
                    "Region_Code": form.Region_Code,
                    "Previously_Insured": form.Previously_Insured,
                    "Vehicle_Age": form.Vehicle_Age,
                    "Vehicle_Damage": form.Vehicle_Damage,
                    "Annual_Premium": form.Annual_Premium,
                    "Policy_Sales_Channel": form.Policy_Sales_Channel,
                    "Vintage": form.Vintage
                }
            },
        )
        
    except Exception as e:
        logging.error(f"Error in prediction route: {e}")
        return templates.TemplateResponse(
            "vehicledata.html",
            {
                "request": request, 
                "context": f"Error: {str(e)}",
                "prediction": None
            },
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Vehicle Insurance Prediction API is running"}

# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)