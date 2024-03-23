from server import HttpServer



def run():

    httpServer = HttpServer.create_app()
    httpServer.run()


if __name__ == '__main__':
    run()