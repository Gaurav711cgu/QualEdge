import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.endpoints import router

app = FastAPI(
    title="Snapdragon Edge AI Console Backend",
    description="FastAPI service serving AIMET compression suite metrics and Q2 hybrid router logic.",
    version="1.0.0"
)

# Set up CORS middleware to allow cross-origin requests from the React + Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local developer environment simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount endpoints
app.include_router(router)

# Serve React frontend static files if they are built (production packaging)
from fastapi.staticfiles import StaticFiles
import os
import sys

frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend", "dist")
is_testing = "pytest" in sys.modules or (len(sys.argv) > 0 and "pytest" in sys.argv[0])

if os.path.exists(frontend_dist) and not is_testing:
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    @app.get("/")
    def read_root():
        return {
            "status": "online",
            "service": "Snapdragon Edge AI Console API",
            "documentation": "/docs"
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
