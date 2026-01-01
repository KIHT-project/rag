from fastapi import FastAPI
from config.logging_config import setup_logging, MDCMiddleware
import logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(MDCMiddleware)

@app.get("/api/v1/hello")
async def getHello():
    logger.info("Hello world log")
    return {"message": "Hello FastAPI"}
