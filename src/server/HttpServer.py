from fastapi import FastAPI
from server.routes import VideoHttpRoute
def create_app():
    app = FastAPI()

    app.include_router(VideoHttpRoute.router)

    return app