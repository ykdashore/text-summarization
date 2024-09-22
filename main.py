from fastapi import FastAPI
from api.endpoints.summarize import router as summarize_router

app = FastAPI()
print(summarize_router)
app.include_router(summarize_router, prefix="/api")



