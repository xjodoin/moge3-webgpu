"""Serve the built static demo locally without compressing multi-GB ONNX files."""
import argparse,functools
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
parser=argparse.ArgumentParser();parser.add_argument('--directory',default='dist');parser.add_argument('--port',type=int,default=5181);args=parser.parse_args()
class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy','same-origin')
        self.send_header('Cross-Origin-Embedder-Policy','require-corp')
        self.send_header('Cross-Origin-Resource-Policy','cross-origin')
        super().end_headers()
ThreadingHTTPServer(('127.0.0.1',args.port),functools.partial(Handler,directory=args.directory)).serve_forever()
