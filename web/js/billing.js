/* Billing helpers for Stripe checkout */
(function () {
    const API_BASE = window.API_BASE_URL || '';

    async function postJSON(url, payload) {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'ngrok-skip-browser-warning': 'true'
            },
            body: JSON.stringify(payload || {})
        });
        if (!response.ok) {
            const text = await response.text();
            throw new Error(text || response.statusText);
        }
        return response.json();
    }

    async function startCheckout(plan, options) {
        const payload = Object.assign({ plan }, options || {});
        const url = `${API_BASE}/api/billing/create-checkout-session`;
        const { url: checkoutUrl, error } = await postJSON(url, payload);
        if (error) throw new Error(error);
        window.location.href = checkoutUrl;
    }

    async function openPortal(sessionId, customerId) {
        const url = `${API_BASE}/api/billing/create-portal-session`;
        const payload = {};
        if (sessionId) payload.session_id = sessionId;
        if (customerId) payload.customer_id = customerId;
        const { url: portalUrl, error } = await postJSON(url, payload);
        if (error) throw new Error(error);
        window.location.href = portalUrl;
    }

    window.Billing = {
        startCheckout,
        openPortal
    };
})();
