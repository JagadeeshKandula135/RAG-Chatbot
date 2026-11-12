// Thin REST client. The session id returned by /api/upload is sent on every
// call via the X-Session-ID header so each user gets isolated state.
let sessionId = null;

try {
  sessionId = sessionStorage.getItem("rag_session_id");
} catch {
  /* storage unavailable */
}

function setSession(id) {
  sessionId = id;
  try {
    sessionStorage.setItem("rag_session_id", id);
  } catch {
    /* ignore */
  }
}

async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (sessionId) headers["X-Session-ID"] = sessionId;
  const res = await fetch(path, { ...options, headers });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

export async function uploadPdf(file) {
  const form = new FormData();
  form.append("file", file);
  const data = await request("/api/upload", { method: "POST", body: form });
  setSession(data.session_id);
  return data;
}

export const getSummary = (topic) =>
  request("/api/summary", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic }),
  });

export const sendChat = (message) =>
  request("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
