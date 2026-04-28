SERVER_HOST = '10.184.94.22'
SERVER_PORT = 30069

from flask import Flask, Response
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)


@app.route('/metrics')
def proxy_metrics():
    url = f'http://{SERVER_HOST}:{SERVER_PORT}/metrics'
    try:
        r = requests.get(url, timeout=10)
        return Response(r.text, status=r.status_code, content_type='text/plain; charset=utf-8')
    except requests.exceptions.RequestException as e:
        return Response(f'# proxy error: {e}', status=502, content_type='text/plain')


if __name__ == '__main__':
    print(f'Proxying http://{SERVER_HOST}:{SERVER_PORT}/metrics -> http://localhost:5001/metrics')
    print('Set USE_PROXY: true in vllm_dashboard.html CONFIG to use this proxy.')
    app.run(host='0.0.0.0', port=5001, debug=False)
