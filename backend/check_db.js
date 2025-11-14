import 'dotenv/config';
import pool, { waitForDbReady } from './db.js';

async function main() {
  try {
    await waitForDbReady({ timeoutMs: 20000, intervalMs: 1000 });
    console.log('DB connection OK: SELECT 1');
    process.exit(0);
  } catch (err) {
    console.error('DB connection failed:', err?.message || err);
    process.exit(1);
  } finally {
    try { await pool.end(); } catch {}
  }
}

main();
