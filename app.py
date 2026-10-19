from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import json
import time
from functools import lru_cache

# Enhanced caching and performance
try:
    from src.constants import APP_HOST, APP_PORT
except ImportError:
    APP_HOST = "127.0.0.1"
    APP_PORT = 5000

try:
    from src.pipline.prediction_pipeline import VehicleData, VehicleDataClassifier
    HAS_ML_MODULES = True
except ImportError:
    HAS_ML_MODULES = False

# Initialize FastAPI application with performance optimizations
app = FastAPI(
    title="InsuranceIQ AI", 
    description="⚡ Lightning-Fast AI-Powered Vehicle Insurance Predictor",
    version="3.0.0"
)

# Performance optimizations
PREDICTION_CACHE = {}
MODEL_CACHE = None

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory='templates')

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for training progress
training_progress = {
    "status": "not_started",
    "progress": 0,
    "message": "Training is available through the project pipeline, not this API.",
}

class DataForm:
    """
    Ultra-fast DataForm class with enhanced caching
    """
    def __init__(self, request: Request):
        self.request = request
        self.cache_key = None
        self._cached_data = None

    async def get_vehicle_data(self):
        """Lightning-fast form data processing with intelligent caching"""
        try:
            if self._cached_data:
                return self._cached_data
                
            form = await self.request.form()
            
            # Process with ultra-fast validation
            data = {
                "Gender": self._safe_str(form.get("Gender"), "Male"),
                "Age": self._safe_int(form.get("Age"), 25, 18, 100),
                "Driving_License": self._safe_int(form.get("Driving_License"), 1, 0, 1),
                "Region_Code": self._safe_float(form.get("Region_Code"), 28.0, 0.0, 100.0),
                "Previously_Insured": self._safe_int(form.get("Previously_Insured"), 0, 0, 1),
                "Annual_Premium": self._safe_float(form.get("Annual_Premium"), 2630.0, 0.0, 1000000.0),
                "Policy_Sales_Channel": self._safe_float(form.get("Policy_Sales_Channel"), 26.0, 0.0, 200.0),
                "Vintage": self._safe_int(form.get("Vintage"), 217, 0, 1000),
                "Vehicle_Age": self._safe_str(form.get("Vehicle_Age"), "< 1 Year"),
                "Vehicle_Damage": self._safe_str(form.get("Vehicle_Damage"), "Yes")
            }
            
            self._cached_data = data
            self.cache_key = hash(str(data))
            return data
            
        except Exception as error:
            raise ValueError("Could not process submitted form data.") from error

    def _safe_str(self, value, default):
        """Micro-optimized string conversion"""
        return str(value).strip() if value not in [None, "", "NA", "null"] else default

    def _safe_int(self, value, default, min_val=None, max_val=None):
        """Ultra-fast integer conversion"""
        try:
            num = int(float(value))
            if min_val is not None and num < min_val:
                return min_val
            if max_val is not None and num > max_val:
                return max_val
            return num
        except (ValueError, TypeError):
            return default

    def _safe_float(self, value, default, min_val=None, max_val=None):
        """Ultra-fast float conversion"""
        try:
            num = float(value)
            if min_val is not None and num < min_val: return min_val
            if max_val is not None and num > max_val: return max_val
            return num
        except (ValueError, TypeError):
            return default

# Performance-optimized prediction cache
@lru_cache(maxsize=1000)
def get_cached_prediction(cache_key: str):
    """Ultra-fast prediction caching"""
    return PREDICTION_CACHE.get(cache_key)

def set_cached_prediction(cache_key: str, prediction: dict):
    """Lightning-fast cache setting"""
    PREDICTION_CACHE[cache_key] = prediction
    # Auto-clean cache when it gets too large
    if len(PREDICTION_CACHE) > 1000:
        PREDICTION_CACHE.clear()

# Hyper-optimized API Endpoints
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Lightning-fast main application page render
    """
    return templates.TemplateResponse(
        "vehicledata.html",
        {"request": request, "context": "Ready"}
    )

@app.get("/api/health")
async def health_check():
    return JSONResponse(
        {
            "status": "ok",
            "service": "vehicle-insurance-api",
            "ml_available": HAS_ML_MODULES,
            "version": "1.0.0",
        }
    )

@app.get("/api/train/status")
async def get_training_status():
    """Return the current API training status."""
    return JSONResponse(training_progress)

@app.post("/api/train/start")
async def start_training():
    return JSONResponse(
        {
            "error": "Training is not exposed by this API. Run the project training pipeline locally.",
            "training_available": False,
        },
        status_code=501,
    )

@app.post("/api/predict")
async def predict_api(request: Request):
    if not HAS_ML_MODULES:
        return JSONResponse(
            {"success": False, "error": "Prediction modules are unavailable."},
            status_code=503,
        )

    try:
        data = await request.json()
        required = {
            "Gender",
            "Age",
            "Driving_License",
            "Region_Code",
            "Previously_Insured",
            "Vehicle_Age",
            "Vehicle_Damage",
            "Annual_Premium",
            "Policy_Sales_Channel",
            "Vintage",
        }
        missing = sorted(required.difference(data))
        if missing:
            return JSONResponse(
                {"success": False, "error": f"Missing fields: {', '.join(missing)}"},
                status_code=422,
            )
        vehicle_data = VehicleData(
            Gender=data["Gender"],
            Age=int(data["Age"]),
            Driving_License=int(data["Driving_License"]),
            Region_Code=float(data["Region_Code"]),
            Previously_Insured=int(data["Previously_Insured"]),
            Vehicle_Age=data["Vehicle_Age"],
            Vehicle_Damage=1 if data["Vehicle_Damage"] == "Yes" else 0,
            Annual_Premium=float(data["Annual_Premium"]),
            Policy_Sales_Channel=float(data["Policy_Sales_Channel"]),
            Vintage=int(data["Vintage"]),
        )
        result = VehicleDataClassifier().predict(vehicle_data)
        return JSONResponse({"success": True, **result})
    except (TypeError, ValueError) as error:
        return JSONResponse(
            {"success": False, "error": f"Invalid prediction input: {error}"},
            status_code=422,
        )
    except Exception:
        return JSONResponse(
            {"success": False, "error": "Prediction is unavailable. Verify model artifacts and server logs."},
            status_code=503,
        )

@app.post("/")
async def predict_route(request: Request):
    if not HAS_ML_MODULES:
        return HTMLResponse(
            "Prediction is unavailable because the model pipeline could not be loaded.",
            status_code=503,
        )

    try:
        start_time = time.perf_counter()
        form = DataForm(request)
        form_data = await form.get_vehicle_data()
        result = VehicleDataClassifier().predict(VehicleData(**form_data))
        prediction_value = result.get("prediction")
        status = "Response-Yes" if prediction_value == 1 else "Response-No"
        response_time = (time.perf_counter() - start_time) * 1000
        return templates.TemplateResponse(
            "vehicledata.html",
            {
                "request": request,
                "context": status,
                "prediction_data": {
                    "value": prediction_value,
                    "confidence": result.get("confidence"),
                    "response_time": f"{response_time:.1f}ms",
                    "speed": "Measured",
                },
                "speed_metrics": {"response_time": f"{response_time:.1f}ms", "grade": "Measured"},
            },
        )
    except (TypeError, ValueError) as error:
        return templates.TemplateResponse(
            "vehicledata.html",
            {"request": request, "context": "Invalid input", "error_message": str(error)},
            status_code=422,
        )
    except Exception:
        return HTMLResponse(
            "Prediction is unavailable. Verify model artifacts and server logs.",
            status_code=503,
        )

@app.get("/train")
async def train_route():
    return JSONResponse(
        {"error": "Training is not exposed by this API.", "training_available": False},
        status_code=501,
    )

@app.get("/api/performance")
async def performance_stats():
    return JSONResponse(
        {
            "status": "ok",
            "model_available": HAS_ML_MODULES,
            "prediction_cache_entries": len(PREDICTION_CACHE),
        }
    )

if __name__ == "__main__":
    # Ultra-fast server configuration
    uvicorn.run(
        "app:app", 
        host=APP_HOST, 
        port=APP_PORT, 
        reload=True,
        workers=1,
        access_log=True
    )