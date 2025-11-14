import 'dotenv/config';
import express from 'express';
import pool, { waitForDbReady, verifyDbConnection } from './db.js';

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

app.get('/healthz', (_req, res) => {
  res.status(200).send('ok');
});

app.get('/healthz/db', async (_req, res) => {
  try {
    const ok = await verifyDbConnection();
    if (ok) return res.status(200).json({ status: 'ok' });
    return res.status(500).json({ status: 'failed' });
  } catch (err) {
    return res.status(500).json({ status: 'error', message: err?.message || String(err) });
  }
});

app.get('/db/time', async (_req, res) => {
  try {
    const { rows } = await pool.query('SELECT NOW() as now');
    return res.status(200).json({ now: rows?.[0]?.now });
  } catch (err) {
    return res.status(500).json({ error: err?.message || String(err) });
  }
});

app.get('/db/tables', async (_req, res) => {
  try {
    const { rows } = await pool.query(`
      SELECT table_name
      FROM information_schema.tables
      WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
      ORDER BY table_name
    `);
    return res.status(200).json({ tables: rows.map(r => r.table_name) });
  } catch (err) {
    return res.status(500).json({ error: err?.message || String(err) });
  }
});

async function start() {
  try {
    await waitForDbReady({ timeoutMs: 30000, intervalMs: 1000 });
  } catch (err) {
    console.error('Database not ready:', err?.message || err);
    process.exit(1);
  }

  app.listen(port, () => {
    console.log(`Backend listening on http://localhost:${port}`);
  });
}

start();
