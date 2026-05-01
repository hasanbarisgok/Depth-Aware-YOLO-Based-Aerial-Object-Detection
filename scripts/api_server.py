from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO


DEFAULT_WEIGHTS = Path("models/aod4_total50_best.pt")


class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[int]


class PredictionResponse(BaseModel):
    model_name: str
    image_width: int
    image_height: int
    detections: list[DetectionItem]
    annotated_image_base64: str | None = None
    annotated_image_mime: str | None = None


def encode_jpeg_base64(image_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", image_bgr)
    if not success:
        raise RuntimeError("Failed to encode annotated image.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded image.")
    return image


def normalize_detections(result: Any, class_names: dict[int, str]) -> list[DetectionItem]:
    detections: list[DetectionItem] = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        detections.append(
            DetectionItem(
                class_id=class_id,
                class_name=class_names[class_id],
                confidence=confidence,
                bbox_xyxy=[x1, y1, x2, y2],
            )
        )
    return detections


def create_app(weights_path: Path, allowed_origins: list[str]) -> FastAPI:
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = YOLO(str(weights_path))

    app = FastAPI(
        title="AOD4 Detection API",
        version="1.0.0",
        description="HTTP API for aerial object detection inference using the trained YOLO model.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": weights_path.name}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        image: UploadFile = File(...),
        conf: float = Form(0.25),
        render: bool = Form(True),
    ) -> PredictionResponse:
        if not (0.0 <= conf <= 1.0):
            raise HTTPException(status_code=400, detail="conf must be between 0.0 and 1.0")

        file_bytes = await image.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            image_bgr = decode_uploaded_image(file_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        results = model.predict(source=image_bgr, conf=conf, verbose=False)
        result = results[0]
        detections = normalize_detections(result, model.names)

        annotated_image_base64 = None
        annotated_image_mime = None
        if render:
            annotated_image = result.plot()
            annotated_image_base64 = encode_jpeg_base64(annotated_image)
            annotated_image_mime = "image/jpeg"

        height, width = image_bgr.shape[:2]
        return PredictionResponse(
            model_name=weights_path.name,
            image_width=width,
            image_height=height,
            detections=detections,
            annotated_image_base64=annotated_image_base64,
            annotated_image_mime=annotated_image_mime,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AOD4 FastAPI inference server.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--cors-origin",
        action="append",
        dest="cors_origins",
        default=["*"],
        help="Allowed CORS origin. Pass multiple times for multiple origins.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.weights, args.cors_origins)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
