from flask import Flask
from flask_restful import Api
from server.routes import UploadHttpRoute

def create_app():
    app = Flask(__name__)
    api = Api(app)

    api.add_resource(UploadHttpRoute.Upload, '/upload')
    api.add_resource(UploadHttpRoute.Video, '/video/<string:video_id>')
    return app