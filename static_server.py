"""
Static Website Server for Railway
Serves only the HTML/CSS/JS frontend
All API calls go to your local PC server
"""
from flask import Flask, send_from_directory, send_file
import os

app = Flask(__name__, static_folder='web')

@app.route('/')
def index():
    return send_file('web/index.html')

@app.route('/privacy')
def privacy():
    return send_file('web/privacy.html')

@app.route('/terms')
def terms():
    return send_file('web/terms.html')

@app.route('/web/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)

@app.route('/<path:path>')
def catch_all(path):
    # Serve any other static files from web folder
    if os.path.exists(os.path.join('web', path)):
        return send_from_directory('web', path)
    return send_file('web/index.html')  # Fallback to index

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
