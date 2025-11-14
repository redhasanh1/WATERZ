import pkg from "pg";
const { Pool } = pkg;

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});

export async function verifyDbConnection() {
  const res = await pool.query("SELECT 1 as ok");
  return res?.rows?.[0]?.ok === 1;
}

export async function waitForDbReady({ timeoutMs = 30000, intervalMs = 1000 } = {}) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const ok = await verifyDbConnection();
      if (ok) return true;
    } catch (err) {
      lastError = err;
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error(`DB not ready after ${timeoutMs}ms${lastError ? `: ${lastError.message}` : ''}`);
}

export default pool;
