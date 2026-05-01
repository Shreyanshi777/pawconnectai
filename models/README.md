# Model Notes

Place trained AI models here when upgrading the MVP:

- `animal_detector.pt` for a custom YOLO model
- `health_classifier.pt` for a trained health classifier

The backend currently supports:

1. Real YOLO mode when `ultralytics` and model weights are available
2. Fallback demo mode with mock predictions
