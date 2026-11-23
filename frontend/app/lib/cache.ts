// Browser-side cache for map data: memory first, then IndexedDB (survives reloads), then the network.
// swrJSON shows a cached copy at once and refreshes it in the background when it is older than its TTL
// ("stale-while-revalidate"). Identical requests in flight are shared. Storage failures (private windows,
// blocked storage) just fall back to memory.
import { getJSON } from './api';

type Entry = { t: number; data: unknown };
const mem = new Map<string, Entry>();
const inflight = new Map<string, Promise<unknown>>();
const DB = 'wildfire-ai-cache';
const STORE = 'json';
const MAX_AGE = 3 * 86400 * 1000;
let dbp: Promise<IDBDatabase | null> | null = null;

function db(): Promise<IDBDatabase | null> {
  if (dbp) return dbp;
  dbp = new Promise((resolve) => {
    try {
      const req = indexedDB.open(DB, 1);
      req.onupgradeneeded = () => req.result.createObjectStore(STORE);
      req.onsuccess = () => {
        const d = req.result;
        resolve(d);
        prune(d);
      };
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
  return dbp;
}

function prune(d: IDBDatabase) {
  try {
    const tx = d.transaction(STORE, 'readwrite');
    const cur = tx.objectStore(STORE).openCursor();
    cur.onsuccess = () => {
      const c = cur.result;
      if (!c) return;
      if (Date.now() - ((c.value as Entry)?.t || 0) > MAX_AGE) c.delete();
      c.continue();
    };
  } catch {
    /* ignore */
  }
}

async function idbGet(key: string): Promise<Entry | null> {
  const d = await db();
  if (!d) return null;
  return new Promise((resolve) => {
    try {
      const r = d.transaction(STORE, 'readonly').objectStore(STORE).get(key);
      r.onsuccess = () => resolve((r.result as Entry) || null);
      r.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

async function idbSet(key: string, e: Entry) {
  const d = await db();
  if (!d) return;
  try {
    d.transaction(STORE, 'readwrite').objectStore(STORE).put(e, key);
  } catch {
    /* quota or blocked: memory still has it */
  }
}

function fetchShared(path: string): Promise<unknown> {
  let p = inflight.get(path);
  if (!p) {
    p = getJSON(path).finally(() => inflight.delete(path));
    inflight.set(path, p);
  }
  return p;
}

export type CacheMeta = { fetchedAt: number; cached: boolean };

/** Calls onData with a cached copy (if any) right away, then with fresh data if the copy is older than ttlMs.
 *  onError gets hadCopy=true when a cached copy is on screen. Returns a cancel function. */
export function swrJSON<T = any>(  // eslint-disable-line @typescript-eslint/no-explicit-any
  path: string,
  ttlMs: number,
  onData: (data: T, meta: CacheMeta) => void,
  onError: (err: Error, hadCopy: boolean, copyAt: number | null) => void,
  onSettled?: () => void, // called once nothing more will arrive
): () => void {
  let alive = true;
  (async () => {
    let e = mem.get(path) || null;
    if (!e) {
      e = await idbGet(path);
      if (e) mem.set(path, e);
    }
    if (!alive) return;
    if (e) onData(e.data as T, { fetchedAt: e.t, cached: true });
    if (e && Date.now() - e.t < ttlMs) return;
    try {
      const data = await fetchShared(path);
      const ne = { t: Date.now(), data };
      mem.set(path, ne);
      idbSet(path, ne);
      if (alive) onData(data as T, { fetchedAt: ne.t, cached: false });
    } catch (err) {
      if (alive) onError(err as Error, !!e, e ? e.t : null);
    }
  })().finally(() => alive && onSettled?.());
  return () => {
    alive = false;
  };
}

/** Synchronous peek at the memory cache (used to paint instantly while panning). */
export function peek<T>(path: string): T | null {
  return (mem.get(path)?.data as T) ?? null;
}
