from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.app.core.config import settings

try:
    import torch
    from torch import nn
    from torchvision import models, transforms
except Exception:
    torch = None
    nn = None
    models = None
    transforms = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


SUPPORTED_DOMESTIC_ANIMALS = {
    "dog",
    "cat",
    "rabbit",
    "bird",
    "cow",
}

EXPECTED_SPECIES_CLASSES = ["dog", "cat", "rabbit", "bird", "cow"]
EXPECTED_HEALTH_CLASSES = ["Healthy", "Mild", "Serious"]


@dataclass
class DetectionResult:
    animal_type: str
    confidence: float
    bbox: dict[str, int]
    crop: Image.Image


class AnimalHealthPipeline:
    def __init__(self) -> None:
        self.detector, self.using_custom_detector = self._load_detector()
        self.species_model, self.species_classes, self.species_transform = self._load_species_model()
        self.health_model, self.health_classes, self.health_transform = self._load_health_model()

    def _load_detector(self) -> tuple[Any, bool]:
        if YOLO is None:
            return None, False

        candidate_paths = [
            (Path(settings.project_root / settings.custom_yolo_model_path), True),
            (Path(settings.yolo_model_path), False),
            (Path(settings.project_root / settings.yolo_model_path), False),
        ]
        for model_path, is_custom in candidate_paths:
            try:
                if model_path.exists() or str(model_path) == settings.yolo_model_path:
                    return YOLO(str(model_path)), is_custom
            except Exception:
                continue
        return None, False

    def preprocess_for_detection(self, image_path: Path) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def _load_health_model(self):
        model_path = Path(settings.project_root / settings.health_model_path)
        if not model_path.exists() or torch is None or models is None or transforms is None:
            return None, None, None

        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            class_names = checkpoint.get("classes", EXPECTED_HEALTH_CLASSES)
            normalized = [str(name).strip() for name in class_names]
            if sorted(normalized) != sorted(EXPECTED_HEALTH_CLASSES):
                return None, None, None
            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, len(normalized))
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )
            return model, normalized, transform
        except Exception:
            return None, None, None

    def _load_species_model(self):
        model_path = Path(settings.project_root / settings.species_model_path)
        if not model_path.exists() or torch is None or models is None or transforms is None:
            return None, None, None

        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            class_names = checkpoint.get("classes", EXPECTED_SPECIES_CLASSES)
            normalized = [str(name).strip().lower() for name in class_names]
            if sorted(normalized) != sorted(EXPECTED_SPECIES_CLASSES):
                return None, None, None
            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, len(normalized))
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )
            return model, normalized, transform
        except Exception:
            return None, None, None

    def detect_animals(self, image_path: Path) -> tuple[list[DetectionResult], str | None]:
        image = self.preprocess_for_detection(image_path)

        if self.detector is None:
            return [], "Animal detector model could not be loaded."

        results = self.detector.predict(
            str(image_path),
            verbose=False,
            imgsz=settings.detector_imgsz,
            conf=settings.detector_confidence_threshold,
            max_det=settings.detector_max_detections,
        )
        detections: list[DetectionResult] = []

        for result in results:
            names = result.names
            for box in result.boxes:
                cls_idx = int(box.cls[0].item())
                label = names.get(cls_idx, str(cls_idx)).lower()
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                crop = image.crop((x1, y1, x2, y2))
                label, confidence = self.resolve_species_label(label, confidence, crop)
                if label not in SUPPORTED_DOMESTIC_ANIMALS:
                    continue

                detections.append(
                    DetectionResult(
                        animal_type=label,
                        confidence=confidence,
                        bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        crop=crop,
                    )
                )

        if not detections:
            refined_label, refined_confidence, refined_runner_up = self.classify_species_details(image)
            refined_margin = refined_confidence - refined_runner_up
            fallback_threshold = 0.60 if refined_label == "rabbit" else 0.68
            fallback_margin = 0.08 if refined_label == "rabbit" else 0.12
            if (
                refined_label in SUPPORTED_DOMESTIC_ANIMALS
                and refined_confidence >= fallback_threshold
                and refined_margin >= fallback_margin
            ):
                detections.append(
                    DetectionResult(
                        animal_type=refined_label,
                        confidence=refined_confidence,
                        bbox={"x1": 0, "y1": 0, "x2": image.width, "y2": image.height},
                        crop=image,
                    )
                )

        return detections, None

    def classify_species(self, crop: Image.Image) -> tuple[str | None, float]:
        label, confidence, _ = self.classify_species_details(crop)
        return label, confidence

    def classify_species_details(self, crop: Image.Image) -> tuple[str | None, float, float]:
        if self.species_model is None or self.species_transform is None or torch is None:
            return None, 0.0, 0.0

        try:
            tensor = self.species_transform(crop).unsqueeze(0)
            with torch.no_grad():
                logits = self.species_model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            best_idx = int(np.argmax(probs))
            sorted_probs = np.sort(probs)[::-1]
            second_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
            return self.species_classes[best_idx].lower(), float(probs[best_idx]), second_prob
        except Exception:
            return None, 0.0, 0.0

    def normalize_species_group(self, animal_type: str | None) -> str:
        animal = (animal_type or "").lower()
        if animal == "bird":
            return "bird"
        return animal

    def resolve_species_label(self, detector_label: str, detector_confidence: float, crop: Image.Image) -> tuple[str, float]:
        species_label, species_confidence, species_runner_up = self.classify_species_details(crop)
        detector_group = self.normalize_species_group(detector_label)
        species_group = self.normalize_species_group(species_label)
        species_margin = species_confidence - species_runner_up

        if species_label not in SUPPORTED_DOMESTIC_ANIMALS:
            return detector_label, detector_confidence

        # If the detector misses a supported class entirely, trust a strong species prediction.
        if detector_label not in SUPPORTED_DOMESTIC_ANIMALS and species_confidence >= 0.68 and species_margin >= 0.12:
            return species_label, round(species_confidence, 4)

        # For tricky domestic-animal confusion like cat vs rabbit, prefer the crop classifier when it is clear.
        if species_label != detector_label:
            if species_confidence >= 0.72 and species_margin >= 0.14:
                return species_label, round(species_confidence, 4)
            if detector_confidence < 0.55 and species_confidence >= 0.62 and species_margin >= 0.10:
                return species_label, round(max(detector_confidence, species_confidence), 4)

        # Even when they agree, let the species model sharpen the confidence a bit.
        if species_label == detector_label and species_confidence >= 0.60:
            return detector_label, round(max(detector_confidence, species_confidence), 4)

        return detector_label, detector_confidence

    def preprocess_for_health(self, crop: Image.Image) -> np.ndarray:
        resized = crop.resize((224, 224))
        array = np.asarray(resized).astype("float32") / 255.0
        return np.expand_dims(array, axis=0)

    def normalize_health_species(self, animal_type: str | None) -> str:
        animal = (animal_type or "").lower()
        return animal or "animal"

    def species_threshold_profile(self, animal_type: str | None) -> dict[str, float]:
        animal = self.normalize_health_species(animal_type)
        profiles = {
            "rabbit": {
                "healthy_red_cap": 0.014,
                "healthy_edge_cap": 0.16,
                "mild_patch_red": 0.05,
                "serious_patch_red": 0.12,
                "serious_purple": 0.065,
            },
            "bird": {
                "healthy_red_cap": 0.016,
                "healthy_edge_cap": 0.18,
                "mild_patch_red": 0.06,
                "serious_patch_red": 0.13,
                "serious_purple": 0.07,
            },
            "cow": {
                "healthy_red_cap": 0.018,
                "healthy_edge_cap": 0.17,
                "mild_patch_red": 0.055,
                "serious_patch_red": 0.12,
                "serious_purple": 0.07,
            },
            "dog": {
                "healthy_red_cap": 0.018,
                "healthy_edge_cap": 0.18,
                "mild_patch_red": 0.06,
                "serious_patch_red": 0.14,
                "serious_purple": 0.075,
            },
            "cat": {
                "healthy_red_cap": 0.018,
                "healthy_edge_cap": 0.18,
                "mild_patch_red": 0.06,
                "serious_patch_red": 0.14,
                "serious_purple": 0.075,
            },
        }
        default = {
            "healthy_red_cap": 0.018,
            "healthy_edge_cap": 0.18,
            "mild_patch_red": 0.06,
            "serious_patch_red": 0.14,
            "serious_purple": 0.075,
        }
        return profiles.get(animal, default)

    def patch_severity_features(self, image: np.ndarray) -> dict[str, float]:
        patch_metrics: list[tuple[float, float, float, float]] = []
        height, width, _ = image.shape
        rows = cols = 3
        for row in range(rows):
            for col in range(cols):
                y1 = int((row / rows) * height)
                y2 = int(((row + 1) / rows) * height)
                x1 = int((col / cols) * width)
                x2 = int(((col + 1) / cols) * width)
                patch = image[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                patch_gray = np.mean(patch, axis=2)
                patch_red = float(np.mean((patch[:, :, 0] > 0.60) & (patch[:, :, 1] < 0.38) & (patch[:, :, 2] < 0.38)))
                patch_dark = float(np.mean(patch_gray < 0.18))
                patch_contrast = float(np.std(patch_gray))
                patch_texture = float(np.mean(np.abs(patch[:, :, 0] - patch[:, :, 2]) > 0.34))
                patch_metrics.append((patch_red, patch_dark, patch_contrast, patch_texture))

        if not patch_metrics:
            return {
                "max_patch_red": 0.0,
                "max_patch_dark": 0.0,
                "max_patch_contrast": 0.0,
                "max_patch_texture": 0.0,
            }

        return {
            "max_patch_red": max(item[0] for item in patch_metrics),
            "max_patch_dark": max(item[1] for item in patch_metrics),
            "max_patch_contrast": max(item[2] for item in patch_metrics),
            "max_patch_texture": max(item[3] for item in patch_metrics),
        }

    def injury_signal_features(self, image: np.ndarray) -> dict[str, float]:
        red = image[:, :, 0]
        green = image[:, :, 1]
        blue = image[:, :, 2]
        grayscale = np.mean(image, axis=2)

        # Excess red helps detect visible bleeding/wounds while suppressing brown fur tones.
        excess_red = np.clip(red - np.maximum(green, blue), 0.0, 1.0)
        excess_red_ratio = float(np.mean(excess_red > 0.22))

        # Purple/dark irregularity is a rough bruising proxy.
        purple_ratio = float(np.mean((red > 0.30) & (blue > 0.30) & (green < 0.26)))

        # Approximate edge density to capture torn fur / sharp lesion boundaries.
        gy, gx = np.gradient(grayscale)
        gradient_mag = np.sqrt((gx * gx) + (gy * gy))
        edge_density = float(np.mean(gradient_mag > 0.18))

        # Strong local color disagreement often appears in wounds and exposed tissue.
        channel_spread = float(np.mean(np.max(image, axis=2) - np.min(image, axis=2)))

        return {
            "excess_red_ratio": excess_red_ratio,
            "purple_ratio": purple_ratio,
            "edge_density": edge_density,
            "channel_spread": channel_spread,
        }

    def heuristic_health_assessment(self, crop: Image.Image, animal_type: str | None) -> tuple[str, float, list[str], str]:
        batch = self.preprocess_for_health(crop)
        image = batch[0]
        profile = self.species_threshold_profile(animal_type)
        grayscale = np.mean(image, axis=2)
        red_ratio = float(np.mean((image[:, :, 0] > 0.55) & (image[:, :, 1] < 0.42) & (image[:, :, 2] < 0.42)))
        dark_ratio = float(np.mean(grayscale < 0.18))
        contrast = float(np.std(grayscale))
        saturation_gap = float(np.mean(np.abs(image[:, :, 0] - image[:, :, 1])))
        mean_brightness = float(np.mean(grayscale))
        bright_red_ratio = float(np.mean((image[:, :, 0] > 0.62) & (image[:, :, 1] < 0.34) & (image[:, :, 2] < 0.34)))
        severe_texture_ratio = float(np.mean(np.abs(image[:, :, 0] - image[:, :, 2]) > 0.34))
        patch_features = self.patch_severity_features(image)
        injury_features = self.injury_signal_features(image)
        max_patch_red = patch_features["max_patch_red"]
        max_patch_dark = patch_features["max_patch_dark"]
        max_patch_contrast = patch_features["max_patch_contrast"]
        max_patch_texture = patch_features["max_patch_texture"]
        excess_red_ratio = injury_features["excess_red_ratio"]
        purple_ratio = injury_features["purple_ratio"]
        edge_density = injury_features["edge_density"]
        channel_spread = injury_features["channel_spread"]

        detected_conditions: list[str] = []
        if bright_red_ratio > 0.045:
            detected_conditions.append("Possible open wound or bleeding-like red region")
        if max_patch_red > 0.12:
            detected_conditions.append("Localized intense red injury-like area detected")
        if purple_ratio > 0.045:
            detected_conditions.append("Possible bruising or deep tissue discoloration detected")
        if mean_brightness < 0.24 and contrast < 0.18 and saturation_gap < 0.12:
            detected_conditions.append("Possible weak body, pale appearance, or low-energy posture")
        if dark_ratio > 0.38 and mean_brightness < 0.19 and contrast < 0.14:
            detected_conditions.append("Possible collapse or severe weakness")
        if dark_ratio > 0.30 and mean_brightness > 0.20:
            detected_conditions.append("Possible bruising, dirt-covered fur, or shadow-heavy body region")
        if contrast > 0.28 and severe_texture_ratio > 0.16:
            detected_conditions.append("High texture variation that may indicate torn fur, wounds, or clutter")
        if max_patch_contrast > 0.34 and max_patch_texture > 0.22:
            detected_conditions.append("One body region shows concentrated visual trauma-like texture")
        if max_patch_contrast > 0.30 and edge_density > 0.23 and severe_texture_ratio > 0.17:
            detected_conditions.append("Possible broken limb or severe mobility problem")
        if saturation_gap > 0.22 and mean_brightness > 0.24:
            detected_conditions.append("Possible skin or fur color irregularity")
        if edge_density > 0.22 and channel_spread > 0.20:
            detected_conditions.append("Sharp lesion-like boundary or exposed tissue pattern detected")

        # Healthy override for common false positives such as dark fur, shadows, or low-detail crops.
        if (
            bright_red_ratio < profile["healthy_red_cap"]
            and excess_red_ratio < 0.018
            and purple_ratio < 0.02
            and saturation_gap < 0.16
            and severe_texture_ratio < 0.14
            and edge_density < profile["healthy_edge_cap"]
        ):
            if mean_brightness < 0.24 or (contrast < 0.24 and dark_ratio < 0.34 and channel_spread < 0.18):
                return (
                    "Healthy",
                    round(max(0.78, 0.93 - (contrast * 0.35)), 4),
                    ["No strong visible injury markers detected"],
                    "No major visible injury markers were detected in the image.",
                )

        if (
            max_patch_red > profile["serious_patch_red"]
            or (max_patch_red > 0.10 and max_patch_texture > 0.20)
            or (bright_red_ratio > 0.085)
            or (excess_red_ratio > 0.075 and edge_density > 0.20)
            or (purple_ratio > profile["serious_purple"] and max_patch_contrast > 0.28)
            or (red_ratio > 0.12 and severe_texture_ratio > 0.18)
            or (dark_ratio > 0.36 and contrast > 0.30)
            or (max_patch_dark > 0.52 and max_patch_contrast > 0.32)
            or (edge_density > 0.26 and channel_spread > 0.24 and max_patch_texture > 0.20)
            or (mean_brightness < 0.20 and contrast < 0.15 and saturation_gap < 0.10)
        ):
            return (
                "Serious",
                round(
                    min(
                        0.97,
                        0.70
                        + max_patch_red
                        + (max_patch_texture * 0.22)
                        + (excess_red_ratio * 0.35)
                        + (purple_ratio * 0.25),
                    ),
                    4,
                ),
                detected_conditions or ["Multiple severe visible abnormality signals detected"],
                "Urgent visible warning signs detected. Immediate rescue or veterinary contact is recommended.",
            )
        if (
            max_patch_red > profile["mild_patch_red"]
            or (max_patch_contrast > 0.28 and max_patch_texture > 0.16)
            or (excess_red_ratio > 0.04 and edge_density > 0.16)
            or purple_ratio > 0.04
            or bright_red_ratio > 0.03
            or (red_ratio > 0.06 and severe_texture_ratio > 0.14)
            or (saturation_gap > 0.20 and contrast > 0.24)
            or (dark_ratio > 0.32 and mean_brightness > 0.25)
            or (edge_density > 0.20 and channel_spread > 0.18)
            or (mean_brightness < 0.28 and contrast < 0.20 and saturation_gap < 0.13)
        ):
            return (
                "Mild",
                round(
                    min(
                        0.90,
                        0.53
                        + (max_patch_red * 0.95)
                        + (max_patch_texture * 0.18)
                        + (excess_red_ratio * 0.28)
                        + (purple_ratio * 0.18),
                    ),
                    4,
                ),
                detected_conditions or ["Moderate visible abnormality signals detected"],
                "The animal may have a mild visible injury or discomfort. A vet check is recommended.",
            )
        return (
            "Healthy",
            round(max(0.74, 0.92 - (contrast * 0.45)), 4),
            detected_conditions or ["No strong visible injury markers detected"],
            "No major visible injury markers were detected in the image.",
        )

    def classify_health(self, crop: Image.Image, animal_type: str | None) -> tuple[str, float, list[str], str]:
        heuristic_status, heuristic_confidence, heuristic_conditions, heuristic_alert = self.heuristic_health_assessment(crop, animal_type)
        if self.health_model is None or self.health_transform is None or torch is None:
            return heuristic_status, heuristic_confidence, heuristic_conditions, heuristic_alert

        tensor = self.health_transform(crop).unsqueeze(0)
        with torch.no_grad():
            logits = self.health_model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        best_idx = int(np.argmax(probs))
        model_status = self.health_classes[best_idx]
        model_confidence = float(probs[best_idx])
        model_conditions = self.describe_conditions(model_status, model_confidence)
        model_alert = self.alert_for_status(model_status)

        combined_conditions = list(dict.fromkeys(heuristic_conditions + model_conditions))

        # Do not let the trained model confidently hide visible injury evidence.
        if model_status == "Healthy" and heuristic_status in {"Mild", "Serious"}:
            if heuristic_confidence >= 0.60:
                return heuristic_status, round(min(0.91, max(heuristic_confidence, 0.68)), 4), combined_conditions, heuristic_alert

        # If the model is uncertain, prefer the species-aware visual triage.
        if model_confidence < 0.78:
            return heuristic_status, round(max(heuristic_confidence, model_confidence), 4), combined_conditions, heuristic_alert

        # If model and heuristic are close but disagree, avoid overconfident healthy outputs.
        if model_status == "Healthy" and heuristic_status == "Healthy":
            return "Healthy", round(min(model_confidence, 0.94), 4), combined_conditions, model_alert
        if model_status == heuristic_status:
            return model_status, round(min(model_confidence, 0.96), 4), combined_conditions, model_alert
        if heuristic_status == "Serious" and model_status == "Mild":
            return "Serious", round(min(0.93, max(heuristic_confidence, model_confidence)), 4), combined_conditions, heuristic_alert
        if heuristic_status == "Mild" and model_status == "Healthy":
            return "Mild", round(min(0.86, max(heuristic_confidence, 0.67)), 4), combined_conditions, heuristic_alert
        return model_status, round(min(model_confidence, 0.95), 4), combined_conditions, model_alert

    def describe_conditions(self, status: str, confidence: float) -> list[str]:
        if status == "Serious":
            return [
                "Model suggests a high-risk visible injury pattern",
                f"Classifier confidence is {confidence:.0%}",
            ]
        if status == "Mild":
            return [
                "Model suggests a moderate visible injury or skin abnormality",
                f"Classifier confidence is {confidence:.0%}",
            ]
        return [
            "Model suggests no major visible injury markers",
            f"Classifier confidence is {confidence:.0%}",
        ]

    def alert_for_status(self, status: str) -> str:
        if status == "Serious":
            return "Urgent visible warning signs detected. Immediate rescue or veterinary contact is recommended."
        if status == "Mild":
            return "The animal may have a mild visible injury or discomfort. A vet check is recommended."
        return "No major visible injury markers were detected in the image."

    def severity_rank(self, status: str) -> int:
        if status == "Serious":
            return 2
        if status == "Mild":
            return 1
        return 0

    def analyze_detection(self, detection: DetectionResult) -> dict:
        health_status, health_confidence, detected_conditions, medical_alert = self.classify_health(detection.crop, detection.animal_type)
        return {
            "animal_type": detection.animal_type,
            "detection_confidence": round(detection.confidence, 4),
            "bbox": detection.bbox,
            "health_status": health_status,
            "health_confidence": health_confidence,
            "detected_conditions": detected_conditions,
            "medical_alert": medical_alert,
        }

    def predict(self, image_path: str | Path) -> dict:
        image_path = Path(image_path)
        detections, load_error = self.detect_animals(image_path)

        if load_error:
            return {
                "analysis_status": "analysis_failed",
                "is_animal": False,
                "animal_type": None,
                "detection_confidence": 0.0,
                "bbox": {"x1": None, "y1": None, "x2": None, "y2": None},
                "health_status": "NotApplicable",
                "health_confidence": 0.0,
                "detected_conditions": [load_error],
                "medical_alert": "AI detection model is unavailable.",
                "all_detections": [],
                "other_detections": [],
            }

        if not detections:
            return {
                "analysis_status": "not_an_animal",
                "is_animal": False,
                "animal_type": None,
                "detection_confidence": 0.0,
                "bbox": {"x1": None, "y1": None, "x2": None, "y2": None},
                "health_status": "NotApplicable",
                "health_confidence": 0.0,
                "detected_conditions": ["No supported domestic animal was detected in the image"],
                "medical_alert": "This image does not appear to contain a supported domestic animal.",
                "all_detections": [],
                "other_detections": [],
            }

        analyzed = [self.analyze_detection(detection) for detection in detections]
        primary = max(
            analyzed,
            key=lambda item: (
                self.severity_rank(item["health_status"]),
                item["detection_confidence"],
                item["health_confidence"],
            ),
        )
        other_detections = [item for item in analyzed if item is not primary]
        all_healthy = all(item["health_status"] == "Healthy" for item in analyzed)

        summary_conditions = list(primary["detected_conditions"])
        if all_healthy and len(analyzed) > 1:
            summary_conditions.append(f"Multiple animals detected ({len(analyzed)} total); no strong visible injury markers were detected.")

        return {
            "analysis_status": "animal_detected",
            "is_animal": True,
            "animal_type": primary["animal_type"],
            "detection_confidence": primary["detection_confidence"],
            "bbox": primary["bbox"],
            "health_status": primary["health_status"],
            "health_confidence": primary["health_confidence"],
            "detected_conditions": summary_conditions,
            "medical_alert": primary["medical_alert"],
            "all_detections": analyzed,
            "other_detections": other_detections,
        }


pipeline = AnimalHealthPipeline()
