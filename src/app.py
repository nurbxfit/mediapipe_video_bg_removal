from server import HttpServer
import asyncio
from hypercorn.config import Config
from hypercorn.asyncio import serve
from config.loader import config as env_conf


def run():
    default_port = env_conf['SERVER_PORT']
    httpServer = HttpServer.create_app()
    server_config = Config()
    server_config.bind = [f'0.0.0.0:{default_port}']
    asyncio.run(serve(httpServer,server_config))
    print(f'started server...')

if __name__ == '__main__':
    run()