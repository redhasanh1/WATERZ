// API Configuration
// Dynamically load tunnel URL from server
let API_BASE_URL = '';

// Fetch tunnel URL on page load
(async function loadTunnelURL() {
    try {
        const response = await fetch('/tunnel_url.txt');
        if (response.ok) {
            API_BASE_URL = (await response.text()).trim();
            console.log('✅ Connected to tunnel:', API_BASE_URL);
        } else {
            // Fallback to localhost if tunnel_url.txt not found
            API_BASE_URL = 'http://localhost:9000';
            console.warn('⚠️  Tunnel URL not found, using localhost');
        }
    } catch (error) {
        // Fallback to localhost on error
        API_BASE_URL = 'http://localhost:9000';
        console.error('❌ Failed to load tunnel URL, using localhost:', error);
    }
})();
