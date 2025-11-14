# Stripe Integration Setup

## Environment Variables Required

Add these to your `.env` file and Railway environment:

```
# Stripe API Keys
STRIPE_SECRET_KEY=sk_live_...  # Your Stripe secret key
STRIPE_WEBHOOK_SECRET=whsec_...  # Webhook signing secret (get from Stripe dashboard)
```

## Stripe Price IDs to Update

Update these in `server_production.py` (lines 356-365):

```python
plans = {
    'pro': {
        'price_id': 'price_YOUR_PRO_PRICE_ID',  # Replace with actual price ID from Stripe
        'credits': 50,
        'name': 'Pro Plan - 50 Videos'
    },
    'enterprise': {
        'price_id': 'price_YOUR_ENTERPRISE_PRICE_ID',  # Replace with actual price ID
        'credits': 300,
        'name': 'Enterprise Plan - 300 Videos'
    }
}
```

## Stripe Webhook Setup

1. Go to Stripe Dashboard → Developers → Webhooks
2. Add endpoint: `https://markremoverai.com/api/stripe/webhook`
3. Select event: `checkout.session.completed`
4. Copy the webhook signing secret and add to environment variables

## Features Implemented

### 1. Credits System
- New users get 5 free video credits on signup
- Credits column added to users table
- Credits displayed in navigation bar with color coding:
  - Purple: 3+ credits
  - Orange: 1-2 credits
  - Red: 0 credits

### 2. Pricing Plans
- **Free**: 5 videos (on signup)
- **Pro ($9.99)**: 50 videos @ 10 seconds each
- **Enterprise ($49.99)**: 300 videos @ 10 seconds each

### 3. Stripe Integration
- Checkout endpoint: `/api/stripe/create-checkout`
- Webhook endpoint: `/api/stripe/webhook`
- Auto-adds credits after successful payment
- Redirects to homepage on success/cancel

### 4. UI Updates
- Credits badge in navigation
- "Buy Credits" button in nav
- Updated pricing section with purchase buttons
- Payment success/cancel messages

## Next Steps (TODO)

1. **Add credit deduction logic**: Deduct 1 credit when user starts video processing
2. **Add Stripe package to Railway**: Install stripe==7.0.0 on Railway
3. **Update price IDs**: Replace placeholder price IDs with your actual Stripe price IDs
4. **Configure webhook**: Set up webhook in Stripe dashboard
5. **Test payments**: Use Stripe test mode to verify checkout flow

## Testing

Use Stripe test cards:
- Success: `4242 4242 4242 4242`
- Decline: `4000 0000 0000 0002`

Set expiry to any future date, CVV to any 3 digits.
