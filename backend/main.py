from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.auth_routes import router as auth_router
from backend.api.survey_routes import router as survey_router

app = FastAPI(title="LFS Conversational AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(survey_router)


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok"}
