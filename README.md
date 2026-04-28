# Paw Connect

Paw Connect is a web-based animal health triage app. A user can upload or capture an animal photo, the backend scans it with AI, generates a report, stores the result in SQLite, and shows the report history in a dashboard.

The current app includes:

- animal scan and triage
- rescue and vet contact lookup by location
- location autocomplete while typing
- dashboard CRUD for saved reports
- report download as PDF
- report-only print output
- care-bot chat for guidance

## Main Features

- AI-powered image scanning for supported domestic animals
- Detection bounding boxes and species refinement
- Health triage output such as `Healthy`, `Mild`, or `Serious`
- Rich report generation with:
  - condition summary
  - body condition
  - injury description
  - animal description
  - possible breed guess when confidence allows
- Report history stored in SQLite
- Dashboard with filters, edit, delete, and bulk delete
- Rescue and veterinary contact lookup by typed location or current GPS location
- Location autocomplete similar to map search
- Browser camera upload and file upload support
- Mobile-friendly sidebar and responsive UI
- Care Bot chat for basic animal-care guidance

## Current Supported Animals

The scanner currently supports:

- `dog`
- `cat`
- `rabbit`
- `bird`
- `cow`

Anything outside that list is treated as unsupported or not recognized.

## What This Project Contains

- `frontend/` for the user interface
- `backend/` for the FastAPI server and AI pipeline
- `database/` for the SQLite database file
- `training/` for dataset preparation notes and scripts
- `models/` for model weights and model notes
- `uploads/` for saved scan images

## What To Run

Backend:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn backend.app.main:app --reload
```

Frontend:

- Open the app in your browser after the backend starts.
- If you are serving the frontend through the backend, use the backend URL.

Environment:

- Keep your local `.env` file at the project root.
- Make sure any required API keys or database paths are set there.

Typical local URL:

```text
http://127.0.0.1:8000
```

## Dependencies

Required to run the project:

- Python 3.10 or 3.11
- `pip`
- SQLite
- A modern browser like Chrome or Edge
- Internet access for first-time setup and external location/contact lookup
- `GEOAPIFY_API_KEY` in `.env` for location detection and rescue contacts
- `GOOGLE_MAPS_API_KEY` in `.env` if you want the stronger Google Places autocomplete/contact lookup path

Python packages are listed in `requirements.txt`. Install them with:

```powershell
python -m pip install -r requirements.txt
```

Hardware note:

- A laptop is the easiest setup.
- On a Raspberry Pi, use a 64-bit OS with enough RAM and expect slower AI inference than on a laptop.
- If the ML model files are missing, the app can still run with fallback behavior, but detection quality will be lower.

### Minimum `.env` values

```env
GEOAPIFY_API_KEY=your_key_here
DATABASE_PATH=database/animal_health.db
```

Optional but recommended for better rescue search:

```env
GOOGLE_MAPS_API_KEY=your_google_places_key_here
```

## Run Steps

### 1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Set up `.env`

Create a `.env` file in the project root:

```env
DATABASE_PATH=database/animal_health.db
GEOAPIFY_API_KEY=your_free_geoapify_key_here
```

Optional for better rescue autocomplete and contact lookup:

```env
GOOGLE_MAPS_API_KEY=your_google_places_key_here
```

### 3. Start the backend

```powershell
python -m uvicorn backend.app.main:app --reload
```

### 4. Open the app

Open:

```text
http://127.0.0.1:8000
```

### 5. Check SQLite data

You can inspect the database with:

```powershell
python -c "import sqlite3; conn = sqlite3.connect('database/animal_health.db'); cur = conn.cursor(); cur.execute('SELECT report_id, animal_name, animal_type, health_status, created_at FROM reports ORDER BY created_at DESC LIMIT 10'); print(cur.fetchall())"
```

## Technology Stack

- Frontend: HTML, CSS, JavaScript
- Backend: FastAPI
- Database: SQLite
- Database driver: built-in `sqlite3`
- AI / ML:
  - YOLO-based animal detection
  - species classification model
  - health classification / triage logic
- Image handling: Pillow, OpenCV-related dependencies through the ML stack

## Important Compatibility Notes

This project is tested for the following kind of setup:

- Windows 10 / Windows 11
- Python 3.10 or 3.11
- SQLite 3
- Chrome, Edge, or any modern mobile browser

Recommended environment:

- 8 GB RAM or more
- a working internet connection for the first setup

## How the App Works

1. The user opens the website.
2. The user uploads an animal image or captures it from camera.
3. The backend sends the image through the AI pipeline.
4. The AI pipeline detects the animal and estimates the species.
5. The health logic classifies the case and builds injury text.
6. The backend generates a structured report.
7. The report is stored in SQLite.
8. The dashboard reads the saved reports from SQLite.
9. The user can print or download only the report section.

## AI / ML Workflow

The scanning part uses the backend AI pipeline in:

- [backend/app/services/ai_pipeline.py](backend/app/services/ai_pipeline.py)

The pipeline does this:

- loads a detector if available
- tries to identify a supported domestic animal
- crops the detected animal
- classifies species if a species model is available
- estimates health severity using the available health model or heuristic fallback

If trained model files are not available, the app can still open and run with fallback logic, but the output quality will be lower.

### Supported output levels

- `Healthy`
- `Mild`
- `Serious`
- `NotApplicable` for non-animal or unsupported cases

### Report output notes

- The report includes animal description and injury description text.
- The report can be downloaded as a PDF.
- Printing uses a report-only layout, not the full app screen.

## Database Design

The project uses SQLite only for application data storage.

### Main tables

- `reports`
- `users`
- `rescue_contacts`
- `vet_contacts`

### What is stored

- image path
- animal type
- animal name
- health status
- confidence scores
- bounding box values
- condition summary
- injury and description fields
- rescue request status
- location details
- creation timestamp

### Database connection files

- [backend/app/core/config.py](backend/app/core/config.py)
- [backend/app/db/session.py](backend/app/db/session.py)
- [backend/app/db/init_db.py](backend/app/db/init_db.py)
- [backend/app/services/crud.py](backend/app/services/crud.py)

## No SQLAlchemy

This project does **not** use SQLAlchemy for the database layer.

Instead, the backend uses:

- direct SQLite connectivity
- Python's built-in `sqlite3`
- plain SQL queries

That keeps the database layer simple and easier to explain in a college demo.

## Dataset Sources

The project training notes reference and support multiple animal datasets. These are the main ones:

### 1. Oxford-IIIT Pet

- Used as the free starting point for cat and dog species training
- Download script:
  - [training/download_free_species_datasets.py](training/download_free_species_datasets.py)
- Official source:
  - https://www.robots.ox.ac.uk/~vgg/data/pets/

### 2. iNaturalist

- Used for broader species diversity
- Useful for:
  - rabbit
  - bird
  - cow
  - donkey
  - horse
  - sheep
  - goat
- Download script:
  - [training/download_inat_species_images.py](training/download_inat_species_images.py)
- Official source:
  - https://help.inaturalist.org/en/support/solutions/articles/151000170342-how-can-i-download-data-from-inaturalist-

### 3. Open Images V7

- Recommended for animal detection training and hard negatives
- Referenced in:
  - [training/dataset_sources.md](training/dataset_sources.md)
- Official source:
  - https://storage.googleapis.com/openimages/web/index.html

### 4. COCO

- Recommended as another detection source for general animal classes
- Referenced in:
  - [training/dataset_sources.md](training/dataset_sources.md)
- COCO reference:
  - https://presentations.cocodataset.org/ECCV18/COCO18-Detect-Overview.pdf

### 5. PetWound research dataset

- Referenced for veterinary wound / injury inspiration
- Source paper:
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC10044392/

### 6. Manual rescue / vet images

- Used for the health dataset workflow
- The expected labels are:
  - `Healthy`
  - `Mild`
  - `Serious`

## Dataset Summary for Presentation

If you need a short explanation for your teacher:

- The project uses **Oxford-IIIT Pet** for cat and dog training
- It uses **iNaturalist** and other sources for dataset preparation and experimentation
- It recommends **Open Images V7** and **COCO** for detection training
- It uses manually reviewed rescue or veterinary images for health classification

## Training Folder Structure

### Health dataset

```text
dataset/
  train/
    Healthy/
    Mild/
    Serious/
  val/
    Healthy/
    Mild/
    Serious/
  test/
    Healthy/
    Mild/
    Serious/
```

### Species dataset

```text
species_dataset/
  train/
    dog/
    cat/
    rabbit/
    bird/
    pigeon/
    parrot/
    duck/
    hen/
    cow/
    donkey/
    horse/
    sheep/
    goat/
    mouse/
  val/
  test/
```

### Detection dataset

```text
domestic_detection_dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

## Project Structure

```text
backend/
  app/
    api/
    core/
    db/
    models/
    schemas/
    services/
    utils/
frontend/
database/
models/
training/
uploads/
requirements.txt
Procfile
railway.json
```

## Local Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Set the SQLite database path

Create a `.env` file in the project root if you want to override the database location:

```env
DATABASE_PATH=database/animal_health.db
```

If you skip this, the app uses the default path above.

### 3. Run the backend

```bash
uvicorn backend.app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

## Useful API Endpoints

- `GET /api/health`
- `GET /api/db-health`
- `POST /api/predict`
- `GET /api/reports`
- `GET /api/reports/{report_id}`
- `PUT /api/reports/{report_id}`
- `DELETE /api/reports/{report_id}`
- `DELETE /api/reports`

## How To Run the AI Scanning Demo

1. Start the backend.
2. Open the homepage.
3. Go to the Scan Animal page.
4. Upload an image or capture from camera.
5. Click Analyse Health.
6. The result panel will show the report.
7. The report is saved to SQLite and appears in the dashboard.

## Deployment

This project can be hosted with:

- GitHub for code storage
- Railway or another Python host for the live app
- SQLite for local or bundled database storage

Deployment files included in the repo:

- [Procfile](Procfile)
- [railway.json](railway.json)

## Notes for a Teacher Demo

What you can say:

- The project is a full-stack animal health triage system.
- The frontend is responsive and works on mobile.
- The backend uses FastAPI.
- The database is SQLite.
- The AI pipeline analyses the scan image and saves the report.
- The dashboard shows historical reports from SQLite.

## Limitations

- The quality of scan results depends on the model files available in `models/`.
- If a custom detector or classifier is not present, the pipeline falls back to heuristic logic.
- The project is meant for demo and academic use, not a veterinary diagnosis tool.

## Running Dataset Preparation Scripts

The `training/` folder contains scripts and notes for building stronger datasets.

Examples:

```bash
python training/download_free_species_datasets.py
python training/download_inat_species_images.py --per-species 120
python training/prepare_health_dataset.py --raw-root dataset/raw --output-root dataset/processed/health
python training/prepare_domestic_species_dataset.py
python training/create_detection_dataset_dirs.py
```

## Final Summary

Paw Connect combines:

- AI image scanning
- animal health triage
- SQLite report storage
- historical dashboard
- rescue and vet guidance
- a responsive mobile-friendly UI

It is built to be easy to present, easy to explain, and easy to extend later.
