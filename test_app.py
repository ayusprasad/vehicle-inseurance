from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("vehicledata.html", {"request": request, "context": "Rendering"})

@app.get("/test")
async def test():
    return {"message": "Server is working!"}

if __name__ == "__main__":
    # Use the import string format for reload to work properly
    uvicorn.run("test_app:app", host="127.0.0.1", port=8000, reload=True)