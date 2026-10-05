// Atlas dashboard client.
//
// One reducer (apply) turns events into a store; render functions draw the
// store. A page load or reconnect gets a `snapshot` (the session so far),
// which is replayed through the same reducer with animations off, so a reload
// shows exactly what a long-running tab shows.

import { cardHTML, dial, timerFraction } from "./cards.js";
import { esc, num, seconds, time, msLeft, countdown, wakePhrase, setClockOffset, serverNow } from "./format.js";

const $ = (id) => document.getElementById(id);
const els = {
  root: document.documentElement, app: document.querySelector(".app"),
  session: $("session"), sphere: $("sphere"), stateName: $("state-name"),
  stateDetail: $("state-detail"), running: $("running"), log: $("log"),
  logInner: $("log-inner"), strip: $("strip"),
};

const ACTIVE_STATES = new Set(["wake", "recording", "transcribing", "thinking", "speaking"]);
// voice_loop goes straight back to listening after an error; keep the error
// on screen for a while (or until the next wake) so it can be read.
const ERROR_HOLD_MS = 6000;
const FAILED = {
  llm: "Claude didn’t answer", transcribe: "Couldn’t turn your speech into text",
  record: "Couldn’t read the microphone", speak: "Couldn’t play the answer aloud",
  wake: "Wake word listening stopped", action: "Couldn’t finish that action",
};

// ---------------------------------------------------------------- store
let store;
function freshStore(session = null) {
  return {
    session, mode: null, startedAt: null, endedAt: null, wake: "Hey Jarvis",
    state: "idle", stateSince: serverNow(), stateTurn: null,
    turns: new Map(), actions: new Map(),
    health: {}, lastError: null, level: 0, levelAt: 0,
  };
}
store = freshStore();

const turnOf = (n, ts) => {
  if (!store.turns.has(n)) store.turns.set(n, { n, ts, actionIds: [] });
  return store.turns.get(n);
};

// Returns a list of one-off effects (animations) for live events.
function apply(ev) {
  const d = ev.data, effects = [];
  if (ev.session !== store.session) {
    store = freshStore(ev.session);
    effects.push("reset");
  }
  switch (ev.type) {
    case "session.started":
      store.mode = d.mode; store.startedAt = ev.ts; store.wake = wakePhrase(d.wake_word);
      break;
    case "session.ended":
      store.endedAt = ev.ts; store.state = "idle";
      break;
    case "state":
      if (store.state !== d.state) store.stateSince = Date.parse(ev.ts);
      store.state = d.state; store.stateTurn = d.turn ?? null;
      if (d.turn != null) turnOf(d.turn, ev.ts);
      if (d.state === "wake") effects.push("flare");
      break;
    case "wake.detected": turnOf(d.turn, ev.ts); break;
    case "recording.finished": turnOf(d.turn, ev.ts).recording = d; break;
    case "turn.user": turnOf(d.turn, ev.ts).user = { ...d, ts: ev.ts }; break;
    case "turn.assistant": turnOf(d.turn, ev.ts).assistant = { ...d, ts: ev.ts }; break;
    case "turn.discarded": turnOf(d.turn, ev.ts).discarded = { ...d, ts: ev.ts }; break;
    case "error":
      store.lastError = { ...d, ts: ev.ts };
      if (d.turn != null) turnOf(d.turn, ev.ts).error = store.lastError;
      break;
    case "health": store.health[d.component] = { ...d, ts: ev.ts }; break;
    case "audio.level": store.level = d.rms; store.levelAt = performance.now(); break;
    case "action": {
      const prev = store.actions.get(d.action_id);
      store.actions.set(d.action_id, { ...d, payload: d.payload || {}, ts: ev.ts,
                                       changed: !!prev && prev.status !== d.status });
      if (d.turn != null) {
        const t = turnOf(d.turn, ev.ts);
        if (!t.actionIds.includes(d.action_id)) t.actionIds.push(d.action_id);
      }
      break;
    }
  }
  return effects;
}

// ---------------------------------------------------------------- render: stage
const STAGE = {
  idle: () => ["Listening for “" + store.wake + "”", "Say it, then ask your question."],
  wake: () => ["Heard you", "Go ahead and ask."],
  recording: () => ["Recording", `${elapsed()} s so far. Stops when you pause.`],
  transcribing: () => ["Transcribing", `Turning your words into text, ${elapsed()} s so far.`],
  thinking: () => ["Thinking", `Waiting for Claude’s answer, ${elapsed()} s so far.`],
  speaking: () => ["Speaking", "Reading the answer aloud."],
  error: () => {
    const e = store.lastError || { stage: "llm", message: "unknown error" };
    const next = e.recoverable === false
      ? "Atlas stopped. Restart it from the terminal."
      : `Atlas is listening again, so say “${esc(store.wake)}” to retry.`;
    return ["Something went wrong", `${FAILED[e.stage] || "Something failed"}: ${esc(e.message)}. ${next}`];
  },
  ended: () => ["Atlas is off", "Start it again with python3 voice_loop.py."],
};

function displayState() {
  if (store.endedAt) return "ended";
  if (store.state === "idle" && store.lastError && store.lastError.recoverable !== false
      && serverNow() - Date.parse(store.lastError.ts) < ERROR_HOLD_MS) return "error";
  return store.state;
}

const elapsed = () => `<span class="num" data-elapsed>${seconds(Math.max(0, serverNow() - store.stateSince))}</span>`;

function renderStage() {
  const state = displayState();
  const [name, detail] = (STAGE[state] || STAGE.idle)();
  els.sphere.dataset.state = state;
  if (els.stateName.textContent !== name) els.stateName.textContent = name;
  els.stateDetail.innerHTML = detail;
  renderRunning();
}

function renderRunning() {
  const chips = [];
  for (const a of store.actions.values()) {
    if (a.kind !== "timer.start") continue;
    const p = a.payload;
    if (a.status === "done" && msLeft(p.ends_at) > 0) {
      chips.push(`<div class="chip">${dial(p)}${esc(p.label || a.title)} ` +
                 `<span class="num" data-countdown="${esc(p.ends_at)}">${countdown(msLeft(p.ends_at))}</span></div>`);
    } else if (a.status === "triggered" && serverNow() - Date.parse(a.ts) < 60_000) {
      chips.push(`<div class="chip rang">${dial({ ...p, ends_at: a.ts })}${esc(p.label || a.title)} timer is done</div>`);
    }
  }
  els.running.innerHTML = chips.join("");
}

// ---------------------------------------------------------------- render: log
const turnEls = new Map();   // turn number -> element
const cardEls = new Map();   // action id -> element

function msgRow(who, text, { time: ts, cls = "", extra = "" } = {}) {
  const label = who === "you" ? "You" : "Atlas";
  return `<div class="msg ${who} ${cls}"><span class="who">${label}</span><p>${text}</p>` +
         (ts ? `<time>${esc(time(ts))}</time>` : "") + extra + `</div>`;
}

function latencyLine(t) {
  const parts = [];
  if (t.user?.stt_ms != null) parts.push(`Heard in ${num(seconds(t.user.stt_ms))} s`);
  if (t.assistant?.llm_ms != null) parts.push(`answered in ${num(seconds(t.assistant.llm_ms))} s`);
  if (!parts.length) return "";
  const line = parts.join(", ");
  return `<span class="latency">${line.charAt(0).toUpperCase() + line.slice(1)}</span>`;
}

function turnRows(t) {
  const current = t.n === store.stateTurn && ACTIVE_STATES.has(store.state);
  const rows = [];

  if (t.discarded) {
    const text = t.discarded.reason === "no_speech"
      ? `Didn’t hear a question. Say “${esc(store.wake)}” and try again.`
      : `Only heard background noise. Say “${esc(store.wake)}” and try again.`;
    return msgRow("atlas", text, { time: t.discarded.ts, cls: "muted" });
  }

  if (t.user) rows.push(msgRow("you", esc(t.user.text), { time: t.user.ts }));
  else if (current && (store.state === "wake" || store.state === "recording")) rows.push(msgRow("you", "Listening…", { cls: "pending" }));
  else if (current && store.state === "transcribing") rows.push(msgRow("you", "Transcribing…", { cls: "pending" }));

  if (t.assistant) {
    rows.push(msgRow("atlas", esc(t.assistant.text), { time: t.assistant.ts, extra: latencyLine(t) }));
  } else if (t.error) {
    const next = t.error.recoverable === false ? "Atlas stopped." : "Ask again in a moment.";
    rows.push(msgRow("atlas", `${FAILED[t.error.stage] || "Something failed"}: ${esc(t.error.message)}. ${next}`,
                     { time: t.error.ts, cls: "failed" }));
  } else if (current && store.state === "thinking") {
    rows.push(msgRow("atlas", "Thinking…", { cls: "pending" }));
  }
  return rows.join("");
}

function renderLog(live) {
  const nearBottom = els.log.scrollHeight - els.log.scrollTop - els.log.clientHeight < 60;

  if (store.turns.size === 0) {
    turnEls.clear(); cardEls.clear();
    els.logInner.innerHTML = `
      <div class="hint">
        <h2>Say “${esc(store.wake)}” to start</h2>
        <p>Then ask out loud, the way you’d ask a person. For example:</p>
        <ul>
          <li>“What’s a good side dish for salmon?”</li>
          <li>“Explain compound interest in one sentence.”</li>
        </ul>
        <p class="fine">Your conversation will appear here as you talk.</p>
      </div>`;
    return;
  }
  if (!turnEls.size) els.logInner.innerHTML = "";

  for (const t of [...store.turns.values()].sort((a, b) => a.n - b.n)) {
    let el = turnEls.get(t.n);
    if (!el) {
      el = document.createElement("article");
      el.className = "turn";
      el.innerHTML = `<div class="rows"></div><div class="msg"><div class="actions"></div></div>`;
      els.logInner.appendChild(el);
      turnEls.set(t.n, el);
    }
    const rows = turnRows(t);
    const rowsEl = el.querySelector(".rows");
    if (rowsEl.innerHTML !== rows) rowsEl.innerHTML = rows;

    const actionsEl = el.querySelector(".actions");
    for (const id of t.actionIds) renderCard(store.actions.get(id), actionsEl, live);
  }

  if (live && nearBottom) els.log.scrollTo({ top: els.log.scrollHeight, behavior: reducedMotion() ? "auto" : "smooth" });
  else if (!live) els.log.scrollTop = els.log.scrollHeight;
}

function renderCard(a, container, live) {
  let el = cardEls.get(a.action_id);
  const isNew = !el;
  if (isNew) {
    el = document.createElement("div");
    container.appendChild(el);
    cardEls.set(a.action_id, el);
  }
  const html = cardHTML(a);
  const cls = `action status-${a.status}`;
  if (el.dataset.html === html && el.dataset.status === a.status) return;
  el.dataset.html = html; el.dataset.status = a.status;
  el.className = cls;
  el.innerHTML = html;
  if (live) {
    // event animations: once, then the class comes off
    const anim = isNew ? "arrive" : a.changed ? "changed" : null;
    if (anim) {
      void el.offsetWidth;
      el.classList.add(anim);
      el.addEventListener("animationend", () => el.classList.remove(anim), { once: true });
    }
  }
}

// ---------------------------------------------------------------- render: strip
function healthCell(component, name, detailFn, extra = "") {
  const h = store.health[component];
  let status = h?.status || "idle";
  let detail = h ? detailFn(h) : "Starting…";
  if (store.endedAt) { status = "idle"; detail = h ? `${esc(h.label)}, stopped` : "Stopped"; }
  const full = detail.replace(/<[^>]+>/g, "");   // tooltip for text the strip truncates
  return `<div class="svc ${status}"><span class="dot ${status}"></span>` +
         `<span class="svc-name">${name}</span>${extra}<span class="svc-detail" title="${full}">${detail}</span></div>`;
}

// the wake-word stream is stopped for the whole turn (see voice_loop.py)
const WAKE_PAUSED = { wake: "paused while you talk", recording: "paused while you talk",
                      transcribing: "paused while Atlas answers", thinking: "paused while Atlas answers",
                      speaking: "paused while Atlas answers" };

const failedAt = (h) => `Failed at ${esc(time(h.ts))}: ${esc(h.error || h.detail || "error")}`;

function renderStrip() {
  const wake = healthCell("wake", "Wake word", (h) => h.status === "failing" ? failedAt(h)
    : `${esc(h.label)}, ${WAKE_PAUSED[store.state] || "listening"}`);
  const stt = healthCell("stt", "Speech to text", (h) => {
    if (h.status === "failing") return failedAt(h);
    if (h.latency_ms == null) return `${esc(h.label)}, ${esc(h.detail || "ready")}`;
    return `${esc(h.label)}, last took ${num(seconds(h.latency_ms))} s`;
  });
  const llm = healthCell("llm", "Claude", (h) => {
    if (h.status === "failing") return failedAt(h);
    if (h.latency_ms == null) return `${esc(h.label)}, ${esc((h.detail || "no calls yet").toLowerCase())}`;
    return `Last answer ${num(seconds(h.latency_ms))} s at ${esc(time(h.ts))}`;
  });
  const bars = [4, 6, 8, 10, 12, 14, 16].map((px) => `<i style="height:${px}px"></i>`).join("");
  const mic = healthCell("mic", "Microphone", (h) => h.status === "failing" ? failedAt(h) : esc(h.label),
                         `<span class="meter" id="meter" aria-hidden="true">${bars}</span>`);
  els.strip.innerHTML = wake + stt + llm + mic;
  renderLevel();
}

function renderLevel() {
  // no level for a moment means the mic stream is paused (Atlas is busy)
  const level = performance.now() - store.levelAt < 500 ? store.level : 0;
  els.sphere.style.setProperty("--level", level.toFixed(3));
  const lit = Math.round(Math.min(1, level * 1.6) * 7);
  document.querySelectorAll("#meter i").forEach((bar, i) => bar.classList.toggle("on", i < lit));
}

// ---------------------------------------------------------------- render: header
function renderHeader() {
  if (!connected) {
    els.session.innerHTML = "Can’t reach Atlas. Retrying…";
    return;
  }
  const demo = store.mode === "demo" || store.mode === "replay"
    ? `<span class="tag">${store.mode === "demo" ? "Demo" : "Replay"}</span>` : "";
  const text = store.endedAt ? `Session ended at ${esc(time(store.endedAt))}`
    : store.startedAt ? `Session started ${esc(time(store.startedAt))}` : "Waiting for Atlas…";
  els.session.innerHTML = demo + text;
}

function renderAll(live) {
  renderHeader();
  renderStage();
  renderLog(live);
  renderStrip();
}

// ---------------------------------------------------------------- effects + ticker
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

function runEffects(effects) {
  for (const fx of effects) {
    if (fx === "reset") { turnEls.clear(); cardEls.clear(); els.logInner.innerHTML = ""; }
    if (fx === "flare") {
      els.sphere.classList.remove("flare-go");
      void els.sphere.getBoundingClientRect();
      els.sphere.classList.add("flare-go");
    }
  }
}
els.sphere.addEventListener("animationend", (e) => {
  if (e.animationName === "flare") els.sphere.classList.remove("flare-go");
});

let lastChipKey = "";
setInterval(() => {
  document.querySelectorAll("[data-elapsed]").forEach((el) => {
    el.textContent = seconds(Math.max(0, serverNow() - store.stateSince));
  });
  document.querySelectorAll("[data-countdown]").forEach((el) => {
    el.textContent = countdown(msLeft(el.dataset.countdown));
  });
  document.querySelectorAll(".dial-remain[data-ends-at]").forEach((el) => {
    const frac = timerFraction({ ends_at: el.dataset.endsAt, duration_s: +el.dataset.duration });
    el.setAttribute("stroke-dasharray", `${(frac * 100).toFixed(1)} 100`);
  });
  // drop chips for timers that ran out or finished ringing a while ago
  const chipKey = [...store.actions.values()]
    .filter((a) => a.kind === "timer.start")
    .map((a) => `${a.action_id}:${a.status}:${msLeft(a.payload.ends_at) > 0}:${serverNow() - Date.parse(a.ts) < 60_000}`)
    .join("|");
  if (chipKey !== lastChipKey) { lastChipKey = chipKey; renderRunning(); }
  if (displayState() !== els.sphere.dataset.state) renderStage();   // error hold ran out
  renderLevel();
}, 200);

// ---------------------------------------------------------------- connection
let connected = false;

function connect() {
  const source = new EventSource("/events");

  source.addEventListener("snapshot", (e) => {
    const snap = JSON.parse(e.data);
    setClockOffset(snap.now - Date.now());
    connected = true;
    els.app.classList.remove("offline");
    store = freshStore(snap.session);
    turnEls.clear(); cardEls.clear(); els.logInner.innerHTML = "";
    els.root.classList.add("replaying");
    for (const ev of snap.events) apply(ev);
    for (const a of store.actions.values()) a.changed = false;
    renderAll(false);
    requestAnimationFrame(() => requestAnimationFrame(() => els.root.classList.remove("replaying")));
  });

  source.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    const effects = apply(ev);
    if (ev.type === "audio.level") { renderLevel(); return; }
    runEffects(effects);
    renderAll(true);
  };

  source.onerror = () => {
    // EventSource retries on its own; the next snapshot rebuilds everything
    connected = false;
    els.app.classList.add("offline");
    renderHeader();
  };
}

els.log.addEventListener("scroll", () => els.log.classList.toggle("scrolled", els.log.scrollTop > 4));
renderAll(false);
connect();
