import json
import cv2
import base64
import numpy
import zmq

class NoDefault:
    pass

class Request:
    def __init__(self, data=str()):
        self.data = json.loads(data)

    def getValue(self, name, default=NoDefault()):
        if name in self.data:
            return self.data[name]

        if isinstance(default, NoDefault):
            raise Exception('Missing value for field: ' + name)

        return default

    def getValues(self):
        return self.data

    def getImage(self, name, default=NoDefault()):
        encoded = self.getValue(name, default)
        buf = base64.decodestring(encoded.encode())

        return cv2.imdecode(numpy.frombuffer(buf, dtype=numpy.uint8),
                            cv2.IMREAD_COLOR)

class Response:
    def __init__(self):
        self.data = {}

    def addValue(self, name, value):
        self.data[name] = value

    def addValues(self, values):
        self.data.update(values)

    def addImage(self, name, image, extension=".jpg"):
        encoded = base64.b64encode(cv2.imencode(extension, image)[1])
        self.addValue(name, encoded.decode('utf-8'))

    def toJSON(self):
        return json.dumps(self.data)

class CloudVis:
    def __init__(self, port, host='cloudvis.qut.edu.au'):
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.connect("tcp://{0}:{1}".format(host, port))

    def __del__(self):
        self.socket.close()

    def run(self, process, data={}):
        print ('Waiting')

        while True:
            try:
                req = Request(self.socket.recv().decode('utf-8'))
                resp = Response()

                print('Request received')

                try:
                    process(req, resp, data)
                except Exception as e:
                    resp.addValues({'error': 1, 'message': str(e)})

                self.socket.send(resp.toJSON().encode())
                print('Replied')

            except KeyboardInterrupt:
                print('Shutting down...')
                break
