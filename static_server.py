"""
Static Website Server for Railway
Serves only the HTML/CSS/JS frontend
All API calls go to your local PC server via tunnel
"""
from flask import Flask, send_from_directory, send_file, Response
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

@app.route('/ads.txt')
def ads_txt():
    """Serve ads.txt for Google AdSense"""
    ads_file = os.path.join('web', 'ads.txt')
    if os.path.exists(ads_file):
        return send_file(ads_file, mimetype='text/plain')
    return "google.com, pub-5884276468441861, DIRECT, f08c47fec0942fa0", 200, {'Content-Type': 'text/plain'}

@app.route('/tunnel_url.txt')
def tunnel_url():
    """
    Serve tunnel URL from environment variable
    This tells the frontend where your PC's backend API is
    """
    # Get tunnel URL from Railway environment variable
    env_url = os.getenv('TUNNEL_URL')

    if env_url:
        return Response(env_url, mimetype='text/plain')
    else:
        # No tunnel URL set - return error message
        return Response(
            "ERROR: TUNNEL_URL not set in Railway environment variables",
            mimetype='text/plain',
            status=500
        )

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
    port = int(os.environ.get('PORT', 8080))
    print("=" * 60)
    print("Static Server for Railway")
    print("=" * 60)
    print(f"Starting on port {port}")

    tunnel_url = os.getenv('TUNNEL_URL')
    if tunnel_url:
        print(f"Backend tunnel: {tunnel_url}")
    else:
        print("WARNING: TUNNEL_URL not set!")

    print("=" * 60)
    app.run(host='0.0.0.0', port=port)
