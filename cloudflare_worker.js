/**
 * Cloudflare Worker to proxy Backblaze B2 downloads with authentication
 *
 * Deploy this at: workers.cloudflare.com
 * It will be available at: https://yourworker.workers.dev
 *
 * This enables FREE unlimited downloads from private B2 buckets!
 */

// ============================================================================
// FILL IN YOUR B2 CREDENTIALS HERE
// ============================================================================
const B2_KEY_ID = '00539db5c1104b50000000001';
const B2_APPLICATION_KEY = 'K005VEORbg6RcsRad3jZPr9n4Fp7jWU';
const B2_BUCKET_NAME = 'watermarkz';

// Auth token cache (valid for 24 hours)
let authToken = null;
let authTokenExpiry = 0;
let downloadUrl = null;

addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function getAuthToken() {
  // Check if cached token is still valid
  if (authToken && Date.now() < authTokenExpiry) {
    return authToken;
  }

  // Get new auth token from B2
  const authResponse = await fetch('https://api.backblazeb2.com/b2api/v2/b2_authorize_account', {
    headers: {
      'Authorization': 'Basic ' + btoa(B2_KEY_ID + ':' + B2_APPLICATION_KEY)
    }
  });

  if (!authResponse.ok) {
    throw new Error('B2 authorization failed');
  }

  const authData = await authResponse.json();
  authToken = authData.authorizationToken;
  downloadUrl = authData.downloadUrl;

  // Cache for 23 hours (token valid for 24h)
  authTokenExpiry = Date.now() + (23 * 60 * 60 * 1000);

  return authToken;
}

async function handleRequest(request) {
  const url = new URL(request.url)

  // Get the path from the request
  const filePath = url.pathname.substring(1) // Remove leading /

  if (!filePath) {
    return new Response('Cloudflare B2 Proxy - Specify file path', {
      status: 200,
      headers: { 'content-type': 'text/plain' }
    })
  }

  // Fetch from B2 with authentication
  try {
    // Get auth token
    const token = await getAuthToken();

    // Construct B2 download URL
    const b2Url = `${downloadUrl}/file/${B2_BUCKET_NAME}/${filePath}`;

    console.log(`Fetching from B2: ${b2Url}`);

    const b2Response = await fetch(b2Url, {
      headers: {
        'Authorization': token
      },
      cf: {
        // Cache for 1 hour
        cacheTtl: 3600,
        cacheEverything: true,
      }
    });

    if (!b2Response.ok) {
      return new Response(`File not found: ${filePath}`, {
        status: 404
      })
    }

    // Return the file with proper headers
    const response = new Response(b2Response.body, {
      status: b2Response.status,
      headers: {
        'Content-Type': b2Response.headers.get('Content-Type') || 'application/octet-stream',
        'Cache-Control': 'public, max-age=3600',
        'Access-Control-Allow-Origin': '*', // Allow CORS
      }
    });

    return response;

  } catch (error) {
    return new Response(`Error fetching from B2: ${error.message}`, {
      status: 500
    })
  }
}
