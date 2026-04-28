from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router
from backend.app.core.config import settings
from backend.app.db.init_db import init_db


app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

app.include_router(router, prefix="/api")
app.mount("/uploads", StaticFiles(directory=str(settings.upload_path)), name="uploads")
app.mount("/static", StaticFiles(directory=str(settings.project_root / "frontend")), name="static")


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(settings.project_root / "frontend" / "index.html")
