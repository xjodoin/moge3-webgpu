// One promise per key also prevents concurrent callers from loading twice.
export class SessionCache {
  entries = new Map();
  created = 0;
  hits = 0;
  async get(key, create) {
    if (this.entries.has(key)) {
      this.hits++;
      return this.entries.get(key);
    }
    this.created++;
    const pending = Promise.resolve().then(create);
    this.entries.set(key, pending);
    try { return await pending; }
    catch (error) {
      if (this.entries.get(key) === pending) this.entries.delete(key);
      throw error;
    }
  }
  async clear() {
    const entries = [...this.entries.values()];
    this.entries.clear();
    const sessions = await Promise.allSettled(entries);
    const releases = await Promise.allSettled(sessions.filter(r => r.status === 'fulfilled').map(r => Promise.resolve().then(() => r.value.release())));
    const failed = releases.find(r => r.status === 'rejected');
    if (failed) throw failed.reason;
  }
  snapshot() { return {created:this.created,hits:this.hits,retained:this.entries.size}; }
}
