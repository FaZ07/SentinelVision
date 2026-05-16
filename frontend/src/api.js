const BASE = '/api'

async function j(res) {
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const api = {
  health: () => fetch(`${BASE}/health`).then(j),

  galleryStatus: () => fetch(`${BASE}/gallery/status`).then(j),

  buildGallery: (files) => {
    const fd = new FormData()
    ;[...files].forEach((f) => fd.append('files', f))
    return fetch(`${BASE}/gallery/build`, { method: 'POST', body: fd }).then(j)
  },

  inspect: (file) => {
    const fd = new FormData()
    fd.append('file', file)
    return fetch(`${BASE}/inspect`, { method: 'POST', body: fd }).then(j)
  },

  benchmark: () => fetch(`${BASE}/benchmark`).then(j),

  exportOnnx: () => fetch(`${BASE}/onnx/export`, { method: 'POST' }).then(j),
}
