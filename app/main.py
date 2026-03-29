from fastapi import FastAPI
from app.routers import emails, health

app = FastAPI(
    title="Poste10 API",
    description="API REST fournissant des outils accessibles via HTTP",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(emails.router)
