from __future__ import annotations

import sqlite3

from backend.app.core.config import settings


CREATE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS rescue_contacts (
    rescue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    phone TEXT NOT NULL,
    email TEXT,
    area TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vet_contacts (
    vet_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    address TEXT NOT NULL,
    phone TEXT NOT NULL,
    area TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL DEFAULT 1,
    image_path TEXT NOT NULL,
    analysis_status TEXT NOT NULL DEFAULT 'animal_detected'
        CHECK (analysis_status IN ('animal_detected', 'not_an_animal', 'analysis_failed')),
    animal_type TEXT,
    animal_name TEXT,
    health_status TEXT NOT NULL
        CHECK (health_status IN ('Healthy', 'Mild', 'Serious', 'NotApplicable')),
    confidence_score REAL NOT NULL DEFAULT 0,
    detection_confidence REAL NOT NULL DEFAULT 0,
    bbox_x1 INTEGER,
    bbox_y1 INTEGER,
    bbox_x2 INTEGER,
    bbox_y2 INTEGER,
    guidance TEXT NOT NULL,
    detected_conditions TEXT,
    animal_reports_json TEXT,
    location_name TEXT,
    location_address TEXT,
    rescue_requested INTEGER NOT NULL DEFAULT 0 CHECK (rescue_requested IN (0, 1)),
    rescue_status TEXT NOT NULL DEFAULT 'not_requested',
    location_lat REAL,
    location_long REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET DEFAULT
);

CREATE TABLE IF NOT EXISTS report_activity_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER,
    action_type TEXT NOT NULL,
    action_label TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    details TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (report_id) REFERENCES reports(report_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_reports_animal_type ON reports(animal_type);
CREATE INDEX IF NOT EXISTS idx_reports_health_status ON reports(health_status);
CREATE INDEX IF NOT EXISTS idx_reports_analysis_status ON reports(analysis_status);
CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at);
CREATE INDEX IF NOT EXISTS idx_report_activity_report_id ON report_activity_logs(report_id);
CREATE INDEX IF NOT EXISTS idx_report_activity_action_type ON report_activity_logs(action_type);
CREATE INDEX IF NOT EXISTS idx_report_activity_created_at ON report_activity_logs(created_at);
"""


DEFAULT_RESCUE_CONTACTS = [
    ("City Animal Rescue", "+91-9000000001", "help@cityrescue.org", "Bangalore"),
    ("Street Paws Rescue", "+91-9000000002", "support@streetpaws.org", "Mysore"),
]

DEFAULT_VET_CONTACTS = [
    ("Green Vet Clinic", "12 Lake Road, Bangalore", "+91-9880000001", "Bangalore"),
    ("Care Animal Hospital", "44 Central Street, Mysore", "+91-9880000002", "Mysore"),
]

DEFAULT_USER_ID = 1
DEFAULT_USER_NAME = "Default User"


def connect_target_db() -> sqlite3.Connection:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        settings.database_path,
        timeout=30,
        detect_types=sqlite3.PARSE_DECLTYPES,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def init_db() -> None:
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    with connect_target_db() as connection:
        connection.executescript(CREATE_SCHEMA_SQL)
        ensure_report_animal_name_column(connection)
        seed_default_user(connection)
        seed_default_contacts(connection)
        refresh_views(connection)
        connection.commit()


def ensure_report_animal_name_column(connection: sqlite3.Connection) -> None:
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(reports)")}
    if "animal_name" not in columns:
        connection.execute("ALTER TABLE reports ADD COLUMN animal_name TEXT")


def seed_default_user(connection: sqlite3.Connection) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO users (user_id, name, email) VALUES (?, ?, ?)",
        (DEFAULT_USER_ID, DEFAULT_USER_NAME, None),
    )


def seed_default_contacts(connection: sqlite3.Connection) -> None:
    rescue_count = connection.execute("SELECT COUNT(*) FROM rescue_contacts").fetchone()[0]
    if rescue_count == 0:
        connection.executemany(
            "INSERT INTO rescue_contacts (name, phone, email, area) VALUES (?, ?, ?, ?)",
            DEFAULT_RESCUE_CONTACTS,
        )

    vet_count = connection.execute("SELECT COUNT(*) FROM vet_contacts").fetchone()[0]
    if vet_count == 0:
        connection.executemany(
            "INSERT INTO vet_contacts (name, address, phone, area) VALUES (?, ?, ?, ?)",
            DEFAULT_VET_CONTACTS,
        )


def refresh_views(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DROP VIEW IF EXISTS report_summary;
        CREATE VIEW report_summary AS
        SELECT
            report_id,
            user_id,
            printf('%03d', COALESCE(user_id, 1)) AS user_code,
            COALESCE(NULLIF(TRIM(animal_name), ''), animal_type) AS display_name,
            animal_type,
            health_status,
            rescue_status,
            created_at
        FROM reports;

        DROP VIEW IF EXISTS report_activity_summary;
        CREATE VIEW report_activity_summary AS
        SELECT
            log_id,
            report_id,
            action_type,
            action_label,
            COALESCE(NULLIF(TRIM(new_value), ''), NULLIF(TRIM(old_value), ''), '') AS display_name,
            old_value,
            new_value,
            details,
            created_at
        FROM report_activity_logs;
        """
    )
