# Stripe Integration Setup - One-Time Credits + Subscriptions

## Overview

WatermarkAI now supports **two types of credit purchases**:
1. **One-Time Credit Packs** - No commitment, credits never expire
2. **Monthly Subscriptions** - Recurring credits every month

## Environment Variables Required

Add these to your `.env` file and Railway environment:

```bash
# Stripe API Keys
STRIPE_SECRET_KEY=sk_live_...  # Your Stripe secret key (or sk_test_... for testing)
STRIPE_WEBHOOK_SECRET=whsec_...  # Webhook signing secret (get from Stripe dashboard)

# One-Time Credit Pack Price IDs (from Stripe Dashboard)
STRIPE_PRICE_ID_CREDITS_10=price_...    # 10 credits - $2.99
STRIPE_PRICE_ID_CREDITS_25=price_...    # 25 credits - $6.99
STRIPE_PRICE_ID_CREDITS_100=price_...   # 100 credits - $24.99
STRIPE_PRICE_ID_CREDITS_250=price_...   # 250 credits - $54.99
STRIPE_PRICE_ID_CREDITS_500=price_...   # 500 credits - $99.99

# Monthly Subscription Price IDs (from Stripe Dashboard)
STRIPE_PRICE_ID_STARTER=price_...       # 20 credits/month - $4.99/month
STRIPE_PRICE_ID_PRO=price_...           # 50 credits/month - $9.99/month
STRIPE_PRICE_ID_ENTERPRISE=price_...    # 300 credits/month - $49.99/month
```

## Step-by-Step Setup

### 1. Create Products in Stripe Dashboard

Go to Stripe Dashboard → Products → Create Product

#### One-Time Credit Packs (5 products)

1. **Starter Pack**
   - Name: "10 Credits - Starter Pack"
   - Pricing: One-time payment
   - Price: $2.99 USD
   - Copy the Price ID (starts with `price_`)

2. **Basic Pack**
   - Name: "25 Credits - Basic Pack"
   - Pricing: One-time payment
   - Price: $6.99 USD

3. **Pro Pack** (Best Value)
   - Name: "100 Credits - Pro Pack"
   - Pricing: One-time payment
   - Price: $24.99 USD

4. **Business Pack**
   - Name: "250 Credits - Business Pack"
   - Pricing: One-time payment
   - Price: $54.99 USD

5. **Enterprise Pack**
   - Name: "500 Credits - Enterprise Pack"
   - Pricing: One-time payment
   - Price: $99.99 USD

#### Monthly Subscriptions (3 products)

1. **Starter Plan**
   - Name: "Starter - 20 Credits/Month"
   - Pricing: Recurring
   - Price: $4.99 USD/month

2. **Pro Plan**
   - Name: "Pro - 50 Credits/Month"
   - Pricing: Recurring
   - Price: $9.99 USD/month

3. **Enterprise Plan**
   - Name: "Enterprise - 300 Credits/Month"
   - Pricing: Recurring
   - Price: $49.99 USD/month

### 2. Configure Webhooks

Webhooks notify your server when payments succeed.

1. Go to Stripe Dashboard → Developers → Webhooks
2. Click "Add endpoint"
3. Enter endpoint URL:
   - Production: `https://markremoverai.com/api/billing/webhook`
   - Test: Use [Stripe CLI](https://stripe.com/docs/stripe-cli) for local forwarding
4. Select events to listen for:
   - `checkout.session.completed` - Initial purchase (one-time & subscription)
   - `invoice.payment_succeeded` - Subscription renewals
   - `customer.subscription.updated` - Plan changes
   - `customer.subscription.deleted` - Cancellations
5. Copy the **Webhook Signing Secret** (starts with `whsec_...`)
6. Add to `.env` as `STRIPE_WEBHOOK_SECRET`

### 3. Install Stripe Package

```bash
pip install stripe
```

On Railway, add to `requirements.txt`:
```
stripe>=7.0.0
```

## What's Been Implemented

### Frontend (web/premium.html)
- ✅ One-Time Credit Packs section with 5 packages
- ✅ Monthly Subscription Plans section
- ✅ Green-themed "ONE-TIME" badges
- ✅ Clear pricing per credit displayed
- ✅ "BEST VALUE" badge on 100-credit pack
- ✅ JavaScript `buyCredits()` function

### Backend (server_production.py)
- ✅ `/api/billing/create-checkout-session` endpoint
  - Handles both one-time (`mode=payment`) and subscription (`mode=subscription`)
  - Maps package names to Stripe Price IDs
  - Stores user_id in checkout metadata
- ✅ `/api/billing/webhook` endpoint
  - Verifies webhook signatures
  - Awards credits on `checkout.session.completed`
  - Handles subscription renewals via `invoice.payment_succeeded`
  - Logs subscription updates/cancellations
- ✅ `/api/billing/create-portal-session` endpoint (for subscription management)

### Billing Integration (web/js/billing.js)
- ✅ `Billing.startOneTimeCheckout()` function
- ✅ Passes `mode: 'payment'` for one-time purchases
- ✅ Shares same checkout endpoint as subscriptions

### Credit System
- ⚠️ **TODO**: Implement credit storage/awarding in webhook handler
- The webhook currently logs credit awards but doesn't persist them
- You need to implement `update_user_credits()` function
- See "Implement Credit Awarding" section below

## Pricing Overview

| Package | Credits | Price | $/Credit | Save vs Smallest |
|---------|---------|-------|----------|------------------|
| **Starter Pack** | 10 | $2.99 | $0.30 | - |
| **Basic Pack** | 25 | $6.99 | $0.28 | 7% |
| **Pro Pack** ⭐ | 100 | $24.99 | $0.25 | 17% |
| **Business Pack** | 250 | $54.99 | $0.22 | 27% |
| **Enterprise Pack** | 500 | $99.99 | $0.20 | 33% |

| Subscription | Credits/mo | Price/mo | $/Credit | Best For |
|--------------|------------|----------|----------|----------|
| **Starter** | 20 | $4.99 | $0.25 | Light use |
| **Pro** | 50 | $9.99 | $0.20 | Regular use |
| **Enterprise** | 300 | $49.99 | $0.17 | Heavy use |

## Implement Credit Awarding

The webhook handler needs to persist credits to your database. Here's how:

### Example: JSON File Storage

```python
# Add this function to server_production.py

def update_user_credits(user_id, credits, reason):
    """Award credits to a user and log the transaction"""
    import json
    from datetime import datetime

    users_file = os.path.join(DATA_DIR, 'users.json')

    # Load users
    if os.path.exists(users_file):
        with open(users_file, 'r') as f:
            users = json.load(f)
    else:
        users = {}

    # Get or create user
    user = users.get(user_id, {
        'id': user_id,
        'credits': 0,
        'credit_history': []
    })

    # Add credits
    user['credits'] = user.get('credits', 0) + credits
    user['credit_history'] = user.get('credit_history', [])
    user['credit_history'].append({
        'timestamp': datetime.now().isoformat(),
        'amount': credits,
        'reason': reason,
        'balance_after': user['credits']
    })
    user['updated_at'] = datetime.now().isoformat()

    # Save
    users[user_id] = user
    os.makedirs(os.path.dirname(users_file), exist_ok=True)
    with open(users_file, 'w') as f:
        json.dump(users, f, indent=2)

    print(f"[CREDITS] User {user_id}: +{credits} credits (now {user['credits']})")
    return user['credits']
```

Then update the webhook handler line 5142:
```python
# Replace the TODO comment with:
update_user_credits(user_id, credits_to_add, f"purchase_{key}")
```

### Example: PostgreSQL Storage

If you're using PostgreSQL (from earlier schema files):

```python
def update_user_credits(user_id, credits, reason):
    """Award credits to a user via PostgreSQL"""
    import psycopg2
    from datetime import datetime

    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cur = conn.cursor()

    # Update credits
    cur.execute("""
        UPDATE users
        SET credits = credits + %s,
            updated_at = NOW()
        WHERE id = %s
        RETURNING credits
    """, (credits, user_id))

    new_balance = cur.fetchone()[0]

    # Log transaction
    cur.execute("""
        INSERT INTO credit_transactions (user_id, amount, reason, balance_after, created_at)
        VALUES (%s, %s, %s, %s, NOW())
    """, (user_id, credits, reason, new_balance))

    conn.commit()
    cur.close()
    conn.close()

    print(f"[CREDITS] User {user_id}: +{credits} credits (now {new_balance})")
    return new_balance
```

## Testing

### Test Cards (Stripe Test Mode)

Use these test cards for testing:
- **Success**: `4242 4242 4242 4242`
- **Decline**: `4000 0000 0000 0002`
- **3D Secure**: `4000 0025 0000 3155`

Set expiry to any future date, CVV to any 3 digits, ZIP to any 5 digits.

### Test One-Time Purchase

1. Start server: `python server_production.py`
2. Go to `/premium.html`
3. Click "Buy 10 Credits" (or any package)
4. Sign in if prompted
5. Complete Stripe Checkout with test card
6. Check server logs:
   ```
   [STRIPE-WEBHOOK] Received: checkout.session.completed
   [BILLING] User xxx purchased credits_10: +10 credits
   ```
7. Verify user balance updated

### Test Subscription

1. Go to `/premium.html`
2. Click "Get Starter" (or any plan)
3. Complete Stripe Checkout
4. Check Stripe Dashboard for new subscription
5. Test renewal (use Stripe CLI to trigger `invoice.payment_succeeded`)

### Local Webhook Testing

Use Stripe CLI for local webhook forwarding:

```bash
# Install Stripe CLI
# https://stripe.com/docs/stripe-cli

# Forward webhooks to local server
stripe listen --forward-to localhost:9000/api/billing/webhook

# Trigger test events
stripe trigger checkout.session.completed
stripe trigger invoice.payment_succeeded
```

## Deployment Checklist

### Before Going Live

- [ ] Create all 8 products in Stripe (5 one-time + 3 subscriptions)
- [ ] Copy all Price IDs to `.env` / Railway environment
- [ ] Add `STRIPE_SECRET_KEY` (use `sk_live_...` for production)
- [ ] Configure webhook endpoint in Stripe Dashboard
- [ ] Add `STRIPE_WEBHOOK_SECRET` to environment
- [ ] Install `stripe` package: `pip install stripe`
- [ ] Implement `update_user_credits()` function
- [ ] Test one-time purchase flow end-to-end
- [ ] Test subscription flow end-to-end
- [ ] Verify webhook events are received
- [ ] Test billing portal access for subscriptions

### After Launch

- [ ] Monitor Stripe Dashboard for transactions
- [ ] Check webhook delivery status
- [ ] Verify credit awarding is working
- [ ] Test subscription renewals
- [ ] Monitor for failed payments

## Troubleshooting

### Checkout fails with "Invalid price ID"

- Verify Price IDs in `.env` match Stripe Dashboard
- Check that `STRIPE_PRICE_ID_CREDITS_10` (etc.) are set correctly
- Ensure products are in the same mode (test vs live) as your API key

### Webhook not receiving events

- Check webhook URL is publicly accessible (use ngrok/Railway URL)
- Verify webhook secret in `.env` matches Stripe Dashboard
- Use Stripe CLI for local testing: `stripe listen`
- Check Stripe Dashboard → Webhooks → Event log for delivery status

### Credits not awarded after purchase

- Check server logs for `[BILLING]` messages
- Verify `update_user_credits()` is implemented
- Ensure webhook includes `client_reference_id` (user_id)
- Check that metadata contains package/plan info

### "Billing is not enabled" error

- Install Stripe: `pip install stripe`
- Check that `STRIPE_ENABLED = True` in server logs
- Verify `STRIPE_SECRET_KEY` is set in `.env`

## User Experience

### Flow for One-Time Purchase

1. User browses premium.html
2. Sees "One-Time Credit Packs" section
3. Clicks "Buy 100 Credits" ($24.99)
4. Redirected to Stripe Checkout
5. Completes payment
6. Redirected to success.html
7. Webhook fires → Credits added instantly
8. User can use credits immediately

### Flow for Subscription

1. User clicks "Get Starter" ($4.99/month)
2. Redirected to Stripe Checkout
3. Completes payment
4. Receives 20 credits immediately
5. Subscription auto-renews monthly
6. Credits replenished on each renewal
7. Can manage subscription via billing portal

## Key Features

✅ **Flexible Purchasing** - One-time OR subscription
✅ **Credits Never Expire** - One-time credits last forever
✅ **No Commitment** - Buy what you need, when you need it
✅ **Volume Discounts** - Bigger packs = lower price per credit
✅ **Subscription Management** - Users can upgrade/downgrade/cancel anytime
✅ **Secure Payments** - All handled by Stripe (PCI compliant)
✅ **Instant Delivery** - Credits awarded immediately via webhook

## Support Resources

- [Stripe Documentation](https://stripe.com/docs)
- [Stripe API Reference](https://stripe.com/docs/api)
- [Stripe Testing Guide](https://stripe.com/docs/testing)
- [Webhook Testing](https://stripe.com/docs/webhooks/test)
- [Stripe CLI](https://stripe.com/docs/stripe-cli)

---

**Last Updated**: 2025-01-14
**Implementation Status**: ✅ Complete (requires credit storage implementation)
