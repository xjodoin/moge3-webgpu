// The public demo keeps large weights in a version-pinned Hugging Face model repo.
// Local development defaults to the exported models in public/models.
let base=(import.meta.env.VITE_MODEL_BASE_URL||'/models').replace(/\/$/,'');
export const modelUrl=path=>`${base}/${path}`;
// New WGSL weights are bundled locally until a model revision contains them.
let sparseBase=(import.meta.env.VITE_SPARSE_MODEL_BASE_URL||'/models').replace(/\/$/,'');
export const sparseModelUrl=path=>`${sparseBase}/${path}`;

export function configureModelLocations({modelBaseUrl,sparseModelBaseUrl}={}){if(modelBaseUrl)base=modelBaseUrl.replace(/\/$/,'');if(sparseModelBaseUrl)sparseBase=sparseModelBaseUrl.replace(/\/$/,'');}
