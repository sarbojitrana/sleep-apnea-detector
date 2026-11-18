"""Local/gateway API. No automatic training or storage of health inputs."""
from contextlib import asynccontextmanager
import hmac
import os
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.pipeline import MODEL_PATH, load_model, predict
from src.oximetry import (SENSOR_DISCLAIMER, SENSOR_MODEL_PATH, load_sensor_model,
                          predict_window)
from src.combined import COMBINED_MODEL_PATH, load_combined_model, predict_combined
from src.predictor import PREDICTOR_PATH, load_predictor, predict as predict_current

MAX_REQUEST_BYTES = 16_384


class ScreeningInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    gender: Literal["Female", "Male"]
    age: int = Field(ge=18, le=100)
    sleep_duration: float = Field(ge=0.5, le=16, description="Hours per night")
    bmi: Literal["Normal", "Overweight", "Obese"]
    heart_rate: float = Field(ge=30, le=220, description="Resting pulse in bpm, measured optically")
    daily_steps: int = Field(ge=0, le=100000)
    loud_snoring: bool = False
    witnessed_pauses: bool = False
    daytime_sleepiness: bool = False


class OximetryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    spo2_samples: list[float | None] = Field(min_length=60, max_length=60,
        description="Exactly 60 consecutive one-second optical oxygen percentages; null means a missing reading.")


class CombinedInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    spo2: float = Field(ge=1, le=100)
    heart_rate: float = Field(ge=30, le=220)
    respiratory_rate: float = Field(ge=1, le=80)
    perfusion_index: float = Field(ge=0.01, le=30)


class PredictorInput(CombinedInput):
    spo2_min: float = Field(ge=1, le=100)
    spo2_std: float = Field(ge=0, le=50)
    sleep_duration: float | None = Field(default=None, ge=0.5, le=16,
        description="Context only; not a trained feature.")


async def validated_body(request, schema):
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        raise HTTPException(415, "Use application/json.")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_REQUEST_BYTES:
            raise HTTPException(413, "Request exceeds 16 KB.")
    try:
        return schema.model_validate_json(body)
    except ValidationError as error:
        details = [{"field": ".".join(map(str, e["loc"])), "message": e["msg"]}
                   for e in error.errors()]
        raise HTTPException(422, detail=details) from error


def create_app(model_path=None, api_key=None, sensor_model_path=None, combined_model_path=None, predictor_path=None):
    bearer = HTTPBearer(auto_error=False)

    @asynccontextmanager
    async def lifespan(application):
        key = api_key if api_key is not None else os.environ.get("APNEA_API_KEY", "")
        if len(key) < 24:
            raise RuntimeError("Set APNEA_API_KEY to a random secret of at least 24 characters.")
        application.state.api_key = key
        legacy_path = Path(model_path or os.environ.get("APNEA_MODEL_PATH", MODEL_PATH))
        application.state.bundle = load_model(legacy_path) if legacy_path.is_file() else None
        sensor_path = Path(sensor_model_path or os.environ.get("APNEA_SENSOR_MODEL_PATH", SENSOR_MODEL_PATH))
        application.state.sensor_bundle = load_sensor_model(sensor_path) if sensor_path.is_file() else None
        combined_path = Path(combined_model_path or os.environ.get("APNEA_COMBINED_MODEL_PATH", COMBINED_MODEL_PATH))
        application.state.combined_bundle = load_combined_model(combined_path) if combined_path.is_file() else None
        current_path = Path(predictor_path or os.environ.get("APNEA_PREDICTOR_PATH", PREDICTOR_PATH))
        application.state.predictor = load_predictor(current_path) if current_path.is_file() else None
        if application.state.bundle is None and application.state.combined_bundle is None and application.state.predictor is None:
            raise RuntimeError("No screening model is installed. Run python predictor.py train.")
        yield

    application = FastAPI(title="Sleep Apnea Screening Prototype", version="4.0.0", lifespan=lifespan,
                          description="Research screening only; no apnea diagnosis or real-time event detection.")

    def authenticate(request: Request,
                     credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if credentials is None or not hmac.compare_digest(
                credentials.credentials.encode(), request.app.state.api_key.encode()):
            raise HTTPException(401, "Valid bearer token required.", headers={"WWW-Authenticate": "Bearer"})

    @application.get("/health")
    def health(request: Request):
        bundle = request.app.state.predictor or request.app.state.combined_bundle or request.app.state.bundle
        return {"status": "ready", "model_id": bundle["metadata"]["model_id"]}

    def legacy_bundle(request):
        if request.app.state.bundle is None:
            raise HTTPException(503, "Legacy lifestyle model is not installed.")
        return request.app.state.bundle

    @application.get("/v1/model", dependencies=[Depends(authenticate)])
    def model_card(request: Request):
        return legacy_bundle(request)["metadata"]

    @application.post("/v1/screen", dependencies=[Depends(authenticate)], openapi_extra={
        "requestBody": {"required": True, "content": {
            "application/json": {"schema": ScreeningInput.model_json_schema()}}}
    })
    async def screen(request: Request):
        record = await validated_body(request, ScreeningInput)
        return predict(legacy_bundle(request), record.model_dump())

    def combined_bundle(request):
        if request.app.state.combined_bundle is None:
            raise HTTPException(503, "Combined screening model is not installed.")
        return request.app.state.combined_bundle

    @application.get("/v2/model", dependencies=[Depends(authenticate)])
    def combined_model_card(request: Request):
        return combined_bundle(request)["metadata"]

    @application.post("/v2/screen", dependencies=[Depends(authenticate)], openapi_extra={
        "requestBody": {"required": True, "content": {
            "application/json": {"schema": CombinedInput.model_json_schema()}}}
    })
    async def combined_screen(request: Request):
        record = await validated_body(request, CombinedInput)
        return predict_combined(combined_bundle(request), record.model_dump())

    def current_predictor(request):
        if request.app.state.predictor is None:
            raise HTTPException(503, "Current predictor is not installed.")
        return request.app.state.predictor

    @application.get("/v3/model", dependencies=[Depends(authenticate)])
    def current_model_card(request: Request):
        return current_predictor(request)["metadata"]

    @application.post("/v3/screen", dependencies=[Depends(authenticate)], openapi_extra={
        "requestBody": {"required": True, "content": {
            "application/json": {"schema": PredictorInput.model_json_schema()}}}
    })
    async def current_screen(request: Request):
        record = await validated_body(request, PredictorInput)
        try:
            return predict_current(current_predictor(request), record.model_dump(exclude_none=True))
        except ValueError as error:
            raise HTTPException(422, detail=str(error)) from error

    def oxygen_bundle(request):
        bundle = request.app.state.sensor_bundle
        if bundle is None:
            raise HTTPException(503, "Oximetry research model is not installed on this service.")
        return bundle

    @application.get("/v1/oximetry/model", dependencies=[Depends(authenticate)])
    def oxygen_model_card(request: Request):
        return oxygen_bundle(request)["metadata"]

    @application.post("/v1/oximetry/screen", dependencies=[Depends(authenticate)], openapi_extra={
        "requestBody": {"required": True, "content": {
            "application/json": {"schema": OximetryInput.model_json_schema()}}}
    })
    async def oxygen_screen(request: Request):
        record = await validated_body(request, OximetryInput)
        bundle = oxygen_bundle(request)
        return {**predict_window(bundle, record.spo2_samples),
                "model_id": bundle["metadata"]["model_id"], "window_seconds": 60,
                "sample_rate_hz": 1, "disclaimer": SENSOR_DISCLAIMER}

    return application


app = create_app()
