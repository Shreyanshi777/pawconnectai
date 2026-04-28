from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from backend.app.core.config import settings


def save_upload_file(upload_file: UploadFile) -> Path:
    suffix = Path(upload_file.filename or "image.jpg").suffix or ".jpg"
    filename = f"{uuid4().hex}{suffix}"
    destination = settings.upload_path / filename

    with destination.open("wb") as buffer:
        buffer.write(upload_file.file.read())

    return destination
