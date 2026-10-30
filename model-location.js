// The public demo keeps large weights in a version-pinned Hugging Face model repo.
// Local development defaults to the exported models in public/models.
const base=(import.meta.env.VITE_MODEL_BASE_URL||'/models').replace(/\/$/,'');
export const modelUrl=path=>`${base}/${path}`;
