# Email-Only Login Setup

Simple email-based authentication with no passwords, no OAuth, no verification - just email → logged in!

## ✅ What's Been Implemented

1. **PostgreSQL users table** (`id`, `email`, `created_at`)
2. **Backend API endpoint** `/api/auth/register`
   - Checks if email exists → returns user
   - If not → creates new user → returns user
   - No password required
3. **Frontend login** (`web/login.html`)
   - Email input connects to API
   - Stores user in `localStorage`
   - Redirects to main app
4. **Route protection** (`web/index.html`)
   - Checks `localStorage` for user
   - Redirects to login if not found

## 🚀 Setup Instructions

### Step 1: Create Database Table

Run this command to create the users table:

```bash
python setup_database.py
```

This will:
- Connect to your PostgreSQL database (using `DATABASE_URL` env var)
- Create the `users` table
- Add email index for fast lookups

### Step 2: Verify Environment Variables

Make sure you have `DATABASE_URL` in your `.env` file:

```
DATABASE_URL=postgresql://username:password@host:port/database
```

### Step 3: Restart Server

Restart `server_production.py` to load the new `/api/auth/register` endpoint:

```bash
python server_production.py
```

### Step 4: Test It!

1. Open `http://localhost:5000/login.html` (or your deployed URL)
2. Enter an email address
3. Click "Sign In" or "Create Account" (both do the same thing!)
4. You'll be logged in and redirected to `index.html`

## 🔒 How It Works

**Login Flow:**
1. User enters email
2. Frontend sends `POST /api/auth/register` with `{email: "user@example.com"}`
3. Backend checks if user exists:
   - **Exists** → Returns existing user (200)
   - **New** → Creates user → Returns new user (201)
4. Frontend stores user in `localStorage`:
   ```javascript
   localStorage.setItem('user', JSON.stringify({
       id: 123,
       email: "user@example.com",
       created_at: "2025-01-01T00:00:00Z"
   }))
   ```
5. Frontend redirects to `index.html`

**Protected Routes:**
- `index.html` checks `localStorage.getItem('user')` on load
- If no user → redirects to `login.html`
- If user exists → allows access

## 📁 Files Created/Modified

- ✅ `create_users_table.sql` - PostgreSQL schema
- ✅ `setup_database.py` - Database setup script
- ✅ `server_production.py` - Added `/api/auth/register` endpoint (line 252)
- ✅ `web/login.html` - Email login JavaScript (line 256-329)
- ✅ `web/index.html` - Auth protection check (line 22-40)

## 🧪 Testing

**Create a user:**
```bash
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com"}'
```

**Response (new user):**
```json
{
  "id": 1,
  "email": "test@example.com",
  "created_at": "2025-01-01T12:00:00",
  "message": "Account created successfully"
}
```

**Response (existing user):**
```json
{
  "id": 1,
  "email": "test@example.com",
  "created_at": "2025-01-01T12:00:00",
  "message": "Logged in successfully"
}
```

## 🔮 Future Enhancements (Optional)

You can add these later if needed:
- Password authentication
- Email verification
- JWT tokens
- Session expiration
- Password reset
- OAuth (Google/GitHub)
- Rate limiting
- CSRF protection

But for now, you have the **EASIEST possible login** - email only! 🎉
