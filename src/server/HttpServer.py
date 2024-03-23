from flask import Flask
from flask_restful import Api
from server.routes import UploadHttpRoute, VideoHttpRoute, ProcessingHttpRoute

def create_app():
    app = Flask(__name__)
    api = Api(app)

    api.add_resource(UploadHttpRoute.Upload, '/upload')
    api.add_resource(VideoHttpRoute.QueryVideo, '/video/<string:video_id>')
    api.add_resource(ProcessingHttpRoute.RemoveBackground, '/remove-bg/video')
    return app