const workerFile = './worker.js';
/** Each client owns one isolated GPU worker and its cached model sessions. */
export class MoGeClient {
  constructor({workerUrl = new URL(workerFile, import.meta.url), WorkerClass = globalThis.Worker} = {}) {
    Object.assign(this, {workerUrl, WorkerClass, worker:null, pending:null, disposed:false, sequence:0});
  }
  static async isSupported() {
    try { return !!(await globalThis.navigator?.gpu?.requestAdapter()); }
    catch { return false; }
  }
  reconstruct(photo, {profile='fast', passes=1, weightMode='exact', signal, onProgress} = {}) {
    if (this.disposed) return Promise.reject(new Error('Client has been disposed'));
    if (this.pending) return Promise.reject(new Error('A reconstruction is already running'));
    if (!(photo instanceof Blob) || !['fast','detail'].includes(profile) || ![1,2,3].includes(passes) || !['exact','compact'].includes(weightMode))
      return Promise.reject(new TypeError('Invalid reconstruction input or options'));
    if (signal?.aborted) return Promise.reject(new DOMException('Reconstruction cancelled','AbortError'));
    return new Promise((resolve,reject) => {
      const id = ++this.sequence;
      const finish = (error,result) => {
        if (this.pending?.id !== id) return;
        signal?.removeEventListener('abort',abort);
        this.pending = null;
        error ? reject(error) : resolve(result);
      };
      const abort = () => {
        this.worker?.terminate(); this.worker = null;
        finish(new DOMException('Reconstruction cancelled','AbortError'));
      };
      this.pending = {id,finish};
      try {
        if (!this.worker) {
          const worker = this.worker = new this.WorkerClass(this.workerUrl,{type:'module',name:'moge-reconstruction'});
          worker.onerror = event => {
            if (this.worker !== worker) return;
            worker.terminate(); this.worker = null;
            this.pending?.finish(new Error(event.message || 'GPU worker failed'));
          };
        }
        this.worker.onmessage = ({data:m}) => {
          if (m.id !== id || this.pending?.id !== id) return;
          if (m.type === 'progress') { try { onProgress?.(m.message); } catch { /* Do not strand inference on a consumer callback error. */ } }
          else if (m.type === 'result') finish(null,m.result);
          else if (m.type === 'error') finish(Object.assign(new Error(m.message),{code:m.code}));
        };
        signal?.addEventListener('abort',abort,{once:true});
        this.worker.postMessage({id,type:'reconstruct',photo,profile,passes,weightMode});
      } catch (error) { this.worker?.terminate(); this.worker=null; finish(error); }
    });
  }
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.worker?.terminate(); this.worker = null;
    this.pending?.finish(new DOMException('Client disposed','AbortError'));
  }
}
