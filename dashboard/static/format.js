// Small formatting helpers shared by app.js and cards.js.

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESC[c]);

// numbers and timings are set in the mono face; units stay in the sans
export const num = (s) => `<span class="num">${esc(s)}</span>`;
export const seconds = (ms) => (ms / 1000).toFixed(1);

// server clock minus browser clock, so countdowns survive a skewed clock
let clockOffset = 0;
export const setClockOffset = (ms) => { clockOffset = ms; };
export const serverNow = () => Date.now() + clockOffset;

const parse = (iso) => (iso instanceof Date ? iso : new Date(iso));

export const time = (iso) =>
  parse(iso).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

export const weekdayShort = (iso) => parse(iso).toLocaleDateString([], { weekday: "short" });
export const dayOfMonth = (iso) => parse(iso).getDate();

export function dayLong(iso) {
  return parse(iso).toLocaleDateString([], { weekday: "long", month: "long", day: "numeric" });
}

// "today at 6:00 PM", "tomorrow at 9:00 AM", "Thursday, October 8 at 3:00 PM"
export function when(iso) {
  const d = parse(iso);
  const today = new Date(serverNow());
  const startOf = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const days = Math.round((startOf(d) - startOf(today)) / 86_400_000);
  const label = days === 0 ? "today" : days === 1 ? "tomorrow" : dayLong(d);
  return `${label} at ${time(d)}`;
}

export function timeRange(startIso, endIso) {
  return endIso ? `${time(startIso)}–${time(endIso)}` : time(startIso);
}

export const msLeft = (endIso) => Math.max(0, parse(endIso).getTime() - serverNow());

export function countdown(ms) {
  const total = Math.ceil(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = String(total % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
}

// "hey_jarvis" -> "Hey Jarvis"
export const wakePhrase = (word) =>
  String(word || "hey_jarvis").split(/[_\s]+/).map((w) => w[0].toUpperCase() + w.slice(1)).join(" ");
