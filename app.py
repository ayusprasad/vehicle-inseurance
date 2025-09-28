from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import VehicleData, VehicleDataClassifier
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
    This class defines the vehicle-related attributes expected from the form.
    """
    def __init__(self, request: Request):
        self.request: Request = request
        self.Gender: str = "Male"
        self.Age: int = 25
        self.Driving_License: int = 1
        self.Region_Code: float = 28.0
        self.Previously_Insured: int = 0
        self.Annual_Premium: float = 2630.0
        self.Policy_Sales_Channel: float = 26.0
        self.Vintage: int = 217
        self.Vehicle_Age: str = "< 1 Year"
        self.Vehicle_Damage: str = "Yes"
                

    async def get_vehicle_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """
        form = await self.request.form()
        self.Gender = str(form.get("Gender")) if form.get("Gender") else "Male"
        self.Age = self._safe_int(form.get("Age"), 25)
        self.Driving_License = self._safe_int(form.get("Driving_License"), 1)
        self.Region_Code = self._safe_float(form.get("Region_Code"), 28.0)
        self.Previously_Insured = self._safe_int(form.get("Previously_Insured"), 0)
        self.Annual_Premium = self._safe_float(form.get("Annual_Premium"), 2630.0)
        self.Policy_Sales_Channel = self._safe_float(form.get("Policy_Sales_Channel"), 26.0)
        self.Vintage = self._safe_int(form.get("Vintage"), 217)
        self.Vehicle_Age = str(form.get("Vehicle_Age")) if form.get("Vehicle_Age") else "< 1 Year"
        self.Vehicle_Damage = str(form.get("Vehicle_Damage")) if form.get("Vehicle_Damage") else "Yes"

    def _safe_int(self, value, default):
        try:
            if value is None or value == "" or value == "NA":
                return default
            return int(value)
        except Exception:
            return default

    def _safe_float(self, value, default):
        try:
            if value is None or value == "" or value == "NA":
                return default
            return float(value)
        except Exception:
            return default

        def _safe_int(self, value, default):
            try:
                if value is None or value == "" or value == "NA":
                    return default
                return int(value)
            except Exception:
                return default

        def _safe_float(self, value, default):
            try:
                if value is None or value == "" or value == "NA":
                    return default
                return float(value)
            except Exception:
                return default

# Route to render the main page with the form
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for vehicle data input.
    """
    return templates.TemplateResponse(
            "vehicledata.html",{"request": request, "context": "Rendering"})

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

# Route to handle form submission and make predictions
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """
    try:
        form = DataForm(request)
        await form.get_vehicle_data()

        # Create VehicleData object with correct arguments
        vehicle_data = VehicleData(
            Gender=form.Gender if form.Gender else "Male",
            Age=form.Age if form.Age is not None else 25,
            Driving_License=form.Driving_License if form.Driving_License is not None else 1,
            Region_Code=form.Region_Code if form.Region_Code is not None else 28.0,
            Previously_Insured=form.Previously_Insured if form.Previously_Insured is not None else 0,
            Vehicle_Age=form.Vehicle_Age if form.Vehicle_Age else "< 1 Year",
            Vehicle_Damage=1 if form.Vehicle_Damage == "Yes" else 0,
            Annual_Premium=form.Annual_Premium if form.Annual_Premium is not None else 2630.0,
            Policy_Sales_Channel=form.Policy_Sales_Channel if form.Policy_Sales_Channel is not None else 26.0,
            Vintage=form.Vintage if form.Vintage is not None else 217
        )

        # Initialize the prediction pipeline
        model_predictor = VehicleDataClassifier()

        # Make a prediction and retrieve the result
        result = model_predictor.predict(vehicle_data)
        value = result.get("prediction", 0)

        # Interpret the prediction result as 'Response-Yes' or 'Response-No'
        status = "Response-Yes" if value == 1 else "Response-No"

        # Render the same HTML page with the prediction result
        return templates.TemplateResponse(
            "vehicledata.html",
            {"request": request, "context": status},
        )

    except Exception as e:
        return {"status": False, "error": f"{e}"}

# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)