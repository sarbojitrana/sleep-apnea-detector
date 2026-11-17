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


def create_app(model_path=None, api_key=None):
    bearer = HTTPBearer(auto_error=False)

    @asynccontextmanager
    async def lifespan(application):
        key = api_key if api_key is not None else os.environ.get("APNEA_API_KEY", "")
        if len(key) < 24:
            raise RuntimeError("Set APNEA_API_KEY to a random secret of at least 24 characters.")
        application.state.api_key = key
        application.state.bundle = load_model(
            Path(model_path or os.environ.get("APNEA_MODEL_PATH", MODEL_PATH)))
        yield

    application = FastAPI(title="Sleep Apnea Screening Prototype", version="2.0.0", lifespan=lifespan,
                          description="Research screening only; no apnea diagnosis or real-time event detection.")

    def authenticate(request: Request,
                     credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if credentials is None or not hmac.compare_digest(
                credentials.credentials.encode(), request.app.state.api_key.encode()):
            raise HTTPException(401, "Valid bearer token required.", headers={"WWW-Authenticate": "Bearer"})

    @application.get("/health")
    def health(request: Request):
        return {"status": "ready", "model_id": request.app.state.bundle["metadata"]["model_id"]}

    @application.get("/v1/model", dependencies=[Depends(authenticate)])
    def model_card(request: Request):
        return request.app.state.bundle["metadata"]

    @application.post("/v1/screen", dependencies=[Depends(authenticate)], openapi_extra={
        "requestBody": {"required": True, "content": {
            "application/json": {"schema": ScreeningInput.model_json_schema()}}}
    })
    async def screen(request: Request):
        if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
            raise HTTPException(415, "Use application/json.")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_REQUEST_BYTES:
                raise HTTPException(413, "Request exceeds 16 KB.")
        try:
            record = ScreeningInput.model_validate_json(body)
        except ValidationError as error:
            # Do not echo health input values in error messages.
            details = [{"field": ".".join(map(str, e["loc"])), "message": e["msg"]}
                       for e in error.errors()]
            raise HTTPException(422, detail=details) from error
        return predict(request.app.state.bundle, record.model_dump())

    return application


app = create_app()
