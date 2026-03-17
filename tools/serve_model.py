#!/usr/bin/env python3
"""
Lightweight local model server.
Run inside a project folder that contains the trained model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
import uvicorn


app = FastAPI(title="Local Model Server")
MODEL = None
TASK = None
TARGET = None
DATASET_PATH = None
FEATURE_COLUMNS: list[str] = []
EXAMPLE_FEATURES: dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str
    task: str | None = None
    target: str | None = None
    dataset_path: str | None = None
    model_loaded: bool
    feature_count: int


class SchemaResponse(BaseModel):
    task: str | None = None
    target: str | None = None
    dataset_path: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    example_features: dict[str, Any] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    prediction: Any
    target: str | None = None
    feature_count: int
    probability: float | None = None
    ignored_extra_features: list[str] | None = None


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _load_dataset_schema(dataset_path: Path | None, target: str | None) -> tuple[list[str], dict[str, Any]]:
    if not dataset_path or not dataset_path.exists():
        return [], {}
    df = pd.read_csv(dataset_path)
    if target and target in df.columns:
        feature_df = df.drop(columns=[target])
    else:
        feature_df = df.copy()
    if feature_df.empty:
        return [], {}
    example_row = feature_df.iloc[0].to_dict()
    example = {str(k): _json_safe(v) for k, v in example_row.items()}
    return feature_df.columns.tolist(), example


def _infer_feature_columns(model: Any, dataset_path: Path | None, target: str | None) -> tuple[list[str], dict[str, Any]]:
    model_features = getattr(model, "feature_names_in_", None)
    dataset_cols, example = _load_dataset_schema(dataset_path, target)
    if model_features is not None:
        cols = [str(c) for c in list(model_features)]
        if example:
            example = {c: example.get(c) for c in cols}
        return cols, example
    return dataset_cols, example


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        task=TASK,
        target=TARGET,
        dataset_path=str(DATASET_PATH) if DATASET_PATH else None,
        model_loaded=MODEL is not None,
        feature_count=len(FEATURE_COLUMNS),
    )


@app.get("/schema", response_model=SchemaResponse)
def schema() -> SchemaResponse:
    return SchemaResponse(
        task=TASK,
        target=TARGET,
        dataset_path=str(DATASET_PATH) if DATASET_PATH else None,
        feature_columns=FEATURE_COLUMNS,
        example_features=EXAMPLE_FEATURES,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse | JSONResponse:
    if MODEL is None:
        return JSONResponse(status_code=500, content={"error": "model not loaded"})
    features = payload.features or {}
    if not isinstance(features, dict):
        return JSONResponse(status_code=400, content={"error": "features must be a JSON object"})

    extra = []
    missing = []
    if FEATURE_COLUMNS:
        missing = [c for c in FEATURE_COLUMNS if c not in features]
        extra = [c for c in features.keys() if c not in FEATURE_COLUMNS]
        if missing:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "missing required features",
                    "missing_features": missing,
                    "expected_features": FEATURE_COLUMNS,
                    "target": TARGET,
                    "example_features": EXAMPLE_FEATURES,
                },
            )
        row = {c: features.get(c) for c in FEATURE_COLUMNS}
    else:
        row = features

    df = pd.DataFrame([row])
    pred = MODEL.predict(df)
    prediction = _json_safe(pred[0])
    out = {
        "prediction": prediction,
        "target": TARGET,
        "feature_count": len(df.columns),
    }
    if TARGET:
        out[TARGET] = prediction
    if extra:
        out["ignored_extra_features"] = extra
    if TASK == "classification" and hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(df)[0]
        out["probability"] = float(max(proba))
    return PredictResponse(**out)


def _install_openapi_examples() -> None:
    example_features = EXAMPLE_FEATURES or {"feature_1": 0}

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version="0.1.0",
            routes=app.routes,
        )
        paths = schema.get("paths", {})
        if "/schema" in paths and "get" in paths["/schema"]:
            paths["/schema"]["get"]["responses"]["200"]["content"]["application/json"]["example"] = {
                "task": TASK,
                "target": TARGET,
                "dataset_path": str(DATASET_PATH) if DATASET_PATH else None,
                "feature_columns": FEATURE_COLUMNS,
                "example_features": example_features,
            }
        if "/predict" in paths and "post" in paths["/predict"]:
            paths["/predict"]["post"]["requestBody"]["content"]["application/json"]["example"] = {
                "features": example_features
            }
            paths["/predict"]["post"]["responses"]["200"]["content"]["application/json"]["example"] = {
                "prediction": 0,
                "target": TARGET,
                "feature_count": len(FEATURE_COLUMNS),
            }
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi_schema = None
    app.openapi = custom_openapi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .joblib model")
    parser.add_argument("--dataset", default=None, help="Path to dataset used for this deployed model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8001, type=int)
    parser.add_argument("--task", default=None)
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    dataset_path = Path(args.dataset).resolve() if args.dataset else None
    if dataset_path and not dataset_path.exists():
        dataset_path = None

    global MODEL, TASK, TARGET, DATASET_PATH, FEATURE_COLUMNS, EXAMPLE_FEATURES
    MODEL = joblib.load(model_path)
    TASK = args.task
    TARGET = args.target
    DATASET_PATH = dataset_path
    FEATURE_COLUMNS, EXAMPLE_FEATURES = _infer_feature_columns(MODEL, dataset_path, TARGET)
    _install_openapi_examples()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
