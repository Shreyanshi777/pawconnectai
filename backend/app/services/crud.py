from __future__ import annotations

from datetime import datetime
import json
from typing import Optional

from backend.app.models.database_models import Report
from backend.app.schemas.report import ReportCreate, ReportUpdate


REPORT_COLUMNS = """
report_id, user_id, image_path, analysis_status, animal_type, animal_name, health_status,
confidence_score, detection_confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
guidance, detected_conditions, animal_reports_json, location_name, location_address,
rescue_requested, rescue_status, location_lat, location_long, created_at
"""

ACTIVITY_COLUMNS = """
log_id, report_id, action_type, action_label, old_value, new_value, details, created_at
"""


def _row_value(row, key: str):
    return row[key]


def _parse_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None


def row_to_report(row) -> Report:
    return Report(
        report_id=_row_value(row, "report_id"),
        user_id=_row_value(row, "user_id"),
        image_path=_row_value(row, "image_path"),
        analysis_status=_row_value(row, "analysis_status"),
        animal_type=_row_value(row, "animal_type"),
        health_status=_row_value(row, "health_status"),
        confidence_score=float(_row_value(row, "confidence_score") or 0),
        detection_confidence=float(_row_value(row, "detection_confidence") or 0),
        bbox_x1=_row_value(row, "bbox_x1"),
        bbox_y1=_row_value(row, "bbox_y1"),
        bbox_x2=_row_value(row, "bbox_x2"),
        bbox_y2=_row_value(row, "bbox_y2"),
        guidance=_row_value(row, "guidance"),
        detected_conditions=_row_value(row, "detected_conditions"),
        animal_reports_json=_row_value(row, "animal_reports_json"),
        location_name=_row_value(row, "location_name"),
        location_address=_row_value(row, "location_address"),
        rescue_requested=bool(_row_value(row, "rescue_requested")),
        rescue_status=_row_value(row, "rescue_status"),
        location_lat=_row_value(row, "location_lat"),
        location_long=_row_value(row, "location_long"),
        animal_name=_row_value(row, "animal_name"),
        created_at=_parse_datetime(_row_value(row, "created_at")),
    )


def row_to_activity_log(row) -> dict:
    return {
        "log_id": _row_value(row, "log_id"),
        "report_id": _row_value(row, "report_id"),
        "action_type": _row_value(row, "action_type"),
        "action_label": _row_value(row, "action_label"),
        "old_value": _row_value(row, "old_value"),
        "new_value": _row_value(row, "new_value"),
        "details": _row_value(row, "details"),
        "created_at": _parse_datetime(_row_value(row, "created_at")),
    }


def report_display_name(report: Report | None) -> str | None:
    if not report:
        return None
    return report.animal_name or report.animal_type


def log_report_activity(
    db,
    *,
    report_id: int | None,
    action_type: str,
    action_label: str,
    old_value: str | None = None,
    new_value: str | None = None,
    details: str | None = None,
) -> None:
    cursor = db.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO report_activity_logs (
                report_id, action_type, action_label, old_value, new_value, details, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                action_type,
                action_label,
                old_value,
                new_value,
                details,
                datetime.utcnow().isoformat(sep=" ", timespec="seconds"),
            ),
        )
        db.commit()
    finally:
        cursor.close()


def create_report(db, report_in: ReportCreate) -> Report:
    user_id = report_in.user_id or 1
    payload = {
        "user_id": user_id,
        "image_path": report_in.image_path,
        "analysis_status": report_in.analysis_status,
        "animal_type": report_in.animal_type,
        "animal_name": report_in.animal_name.strip() if isinstance(report_in.animal_name, str) and report_in.animal_name.strip() else None,
        "health_status": report_in.health_status,
        "confidence_score": report_in.confidence_score,
        "detection_confidence": report_in.detection_confidence,
        "bbox_x1": report_in.bbox.x1,
        "bbox_y1": report_in.bbox.y1,
        "bbox_x2": report_in.bbox.x2,
        "bbox_y2": report_in.bbox.y2,
        "guidance": report_in.guidance,
        "detected_conditions": json.dumps(report_in.detected_conditions),
        "animal_reports_json": json.dumps(report_in.animal_reports),
        "location_name": report_in.location_name,
        "location_address": report_in.location_address,
        "rescue_requested": int(report_in.rescue_requested),
        "rescue_status": report_in.rescue_status,
        "location_lat": report_in.location_lat,
        "location_long": report_in.location_long,
        "created_at": datetime.utcnow().isoformat(sep=" ", timespec="seconds"),
    }
    cursor = db.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO reports (
                user_id, image_path, analysis_status, animal_type, animal_name, health_status,
                confidence_score, detection_confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                guidance, detected_conditions, animal_reports_json, location_name, location_address,
                rescue_requested, rescue_status, location_lat, location_long, created_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
            """,
            (
                payload["user_id"],
                payload["image_path"],
                payload["analysis_status"],
                payload["animal_type"],
                payload["animal_name"],
                payload["health_status"],
                payload["confidence_score"],
                payload["detection_confidence"],
                payload["bbox_x1"],
                payload["bbox_y1"],
                payload["bbox_x2"],
                payload["bbox_y2"],
                payload["guidance"],
                payload["detected_conditions"],
                payload["animal_reports_json"],
                payload["location_name"],
                payload["location_address"],
                payload["rescue_requested"],
                payload["rescue_status"],
                payload["location_lat"],
                payload["location_long"],
                payload["created_at"],
            ),
        )
        db.commit()
        report_id = cursor.lastrowid
        report = get_report(db, report_id)  # type: ignore[assignment]
        log_report_activity(
            db,
            report_id=report_id,
            action_type="insert",
            action_label="Report inserted",
            new_value=report_display_name(report),
            details=f"Inserted report for {report.animal_type if report else 'animal'}",
        )
        return report  # type: ignore[return-value]
    finally:
        cursor.close()


def get_reports(
    db,
    animal_type: Optional[str] = None,
    health_status: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
):
    sql = f"SELECT {REPORT_COLUMNS} FROM reports"
    clauses: list[str] = []
    params: list[object] = []

    if animal_type:
        clauses.append("LOWER(animal_type) = LOWER(?)")
        params.append(animal_type)
    if health_status:
        clauses.append("health_status = ?")
        params.append(health_status)
    if start_date:
        clauses.append("created_at >= ?")
        params.append(start_date.isoformat(sep=" ", timespec="seconds"))
    if end_date:
        clauses.append("created_at <= ?")
        params.append(end_date.isoformat(sep=" ", timespec="seconds"))

    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC, report_id DESC"

    cursor = db.cursor()
    try:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        return [row_to_report(row) for row in rows]
    finally:
        cursor.close()


def get_report(db, report_id: int) -> Optional[Report]:
    cursor = db.cursor()
    try:
        cursor.execute(f"SELECT {REPORT_COLUMNS} FROM reports WHERE report_id = ?", (report_id,))
        row = cursor.fetchone()
        return row_to_report(row) if row else None
    finally:
        cursor.close()


def update_report(db, report_id: int, report_in: ReportUpdate) -> Optional[Report]:
    current_report = get_report(db, report_id)
    if not current_report:
        return None

    fields: list[str] = []
    params: list[object] = []
    activity_parts: list[str] = []
    old_value = current_report.animal_name or current_report.animal_type

    if report_in.rescue_status is not None:
        fields.append("rescue_status = ?")
        params.append(report_in.rescue_status)
        activity_parts.append(f"rescue_status={report_in.rescue_status}")
    if report_in.animal_name is not None:
        animal_name = report_in.animal_name.strip() if isinstance(report_in.animal_name, str) else ""
        fields.append("animal_name = ?")
        params.append(animal_name or None)
        activity_parts.append(f"animal_name={animal_name or 'cleared'}")
        if current_report.animal_reports_json:
            try:
                animal_reports = json.loads(current_report.animal_reports_json)
                if isinstance(animal_reports, list) and animal_reports:
                    first_report = animal_reports[0]
                    if isinstance(first_report, dict):
                        first_report["animal_name"] = animal_name or None
                        fields.append("animal_reports_json = ?")
                        params.append(json.dumps(animal_reports))
            except Exception:
                pass

    if not fields:
        return current_report

    cursor = db.cursor()
    try:
        sql = f"UPDATE reports SET {', '.join(fields)} WHERE report_id = ?"
        cursor.execute(sql, (*params, report_id))
        if cursor.rowcount == 0:
            db.rollback()
            return None
        db.commit()
        updated_report = get_report(db, report_id)
        if updated_report:
            action_type = "rename" if report_in.animal_name is not None and report_in.rescue_status is None else "update"
            action_label = "Report renamed" if action_type == "rename" else "Report updated"
            log_report_activity(
                db,
                report_id=report_id,
                action_type=action_type,
                action_label=action_label,
                old_value=old_value,
                new_value=report_display_name(updated_report),
                details=", ".join(activity_parts) if activity_parts else "Report fields updated",
            )
        return updated_report
    finally:
        cursor.close()


def delete_report(db, report_id: int) -> bool:
    current_report = get_report(db, report_id)
    cursor = db.cursor()
    try:
        cursor.execute("DELETE FROM reports WHERE report_id = ?", (report_id,))
        db.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            log_report_activity(
                db,
                report_id=report_id,
                action_type="delete",
                action_label="Report deleted",
                old_value=report_display_name(current_report),
                new_value=None,
                details="Deleted single report",
            )
        return deleted
    finally:
        cursor.close()


def delete_all_reports(db) -> int:
    cursor = db.cursor()
    try:
        cursor.execute("DELETE FROM reports")
        deleted_count = cursor.rowcount
        db.commit()
        if deleted_count:
            log_report_activity(
                db,
                report_id=None,
                action_type="bulk_delete",
                action_label="All reports deleted",
                details=f"Deleted {deleted_count} report(s) in bulk",
            )
        return deleted_count
    finally:
        cursor.close()


def get_report_activity_logs(db, limit: int = 25):
    cursor = db.cursor()
    try:
        cursor.execute(
            f"SELECT {ACTIVITY_COLUMNS} FROM report_activity_logs ORDER BY created_at DESC, log_id DESC LIMIT ?",
            (limit,),
        )
        return [row_to_activity_log(row) for row in cursor.fetchall()]
    finally:
        cursor.close()
