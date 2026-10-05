// Action cards, keyed by action kind. A new feature adds one renderer here;
// unknown kinds still get the generic card, so nothing breaks in between.
//
// Each renderer returns { icon, what, how, extra? } as HTML strings.

import { esc, num, time, when, dayLong, timeRange, weekdayShort, dayOfMonth, msLeft, countdown } from "./format.js";

// Progress ring used by timer cards and the running-timer chips. app.js's
// ticker updates [data-ends-at] elements every 200ms.
export function dial(payload) {
  const frac = timerFraction(payload);
  return `<svg viewBox="-20 -20 40 40" aria-hidden="true"><circle class="dial-track" r="16"/>` +
    `<circle class="dial-remain" r="16" pathLength="100" stroke-dasharray="${(frac * 100).toFixed(1)} 100" ` +
    `data-ends-at="${esc(payload.ends_at)}" data-duration="${payload.duration_s}"/></svg>`;
}

export function timerFraction(payload) {
  return Math.min(1, msLeft(payload.ends_at) / (payload.duration_s * 1000 || 1));
}

const liveCountdown = (endsAt) => `<span class="num" data-countdown="${esc(endsAt)}">${countdown(msLeft(endsAt))}</span>`;

const failed = (verb, a) => ({ what: `Couldn’t ${verb}`, how: esc(a.error || "Something went wrong.") });

const RENDERERS = {
  "timer.start"(a) {
    const p = a.payload, label = esc(p.label || a.title || "Timer");
    const icon = dial(p);
    switch (a.status) {
      case "pending": return { icon, what: `Starting timer: ${label}`, how: "One moment…" };
      case "done": return { icon, what: `Timer started: ${label}`,
                            how: `${liveCountdown(p.ends_at)} left, rings at ${time(p.ends_at)}` };
      case "triggered": return { icon, what: `Timer done: ${label}`, how: `Rang at ${time(a.ts)}` };
      case "cancelled": return { icon, what: `Timer cancelled: ${label}`, how: "It won’t ring." };
      default: return { icon, ...failed(`start the ${label} timer`, a) };
    }
  },

  "calendar.create"(a) {
    const p = a.payload, title = esc(p.title || a.title);
    const icon = `<div class="date-tile"><small>${esc(weekdayShort(p.start))}</small><b>${dayOfMonth(p.start)}</b></div>`;
    const how = `${esc(dayLong(p.start))}, ${esc(timeRange(p.start, p.end))}`;
    const extra = [p.location, p.calendar && `${p.calendar} calendar`].filter(Boolean).map(esc).join(". ");
    switch (a.status) {
      case "pending": return { icon, what: `Adding to calendar: ${title}`, how };
      case "done": return { icon, what: `Added to calendar: ${title}`, how, extra };
      case "cancelled": return { icon, what: `Removed from calendar: ${title}`, how };
      default: return { icon, ...failed(`add ${title} to your calendar`, a) };
    }
  },

  "reminder.create"(a) {
    const p = a.payload, text = esc(p.text || a.title);
    const icon = `<svg viewBox="-20 -20 40 40" aria-hidden="true"><path d="M0 -15 L4 -4 L15 0 L4 4 L0 15 L-4 4 L-15 0 L-4 -4 Z" fill="none" stroke="#C9A45C" stroke-width="1.6"/><circle r="2.4" fill="#C9A45C"/></svg>`;
    switch (a.status) {
      case "pending": return { icon, what: "Setting a reminder…", how: text };
      case "done": return { icon, what: `Reminder set: ${text}`, how: esc(capitalize(when(p.due_at))) };
      case "triggered": return { icon, what: `Reminder: ${text}`, how: `Due ${esc(when(p.due_at))}` };
      case "cancelled": return { icon, what: `Reminder cancelled: ${text}`, how: "" };
      default: return { icon, ...failed("set that reminder", a) };
    }
  },

  "calendar.list"(a) {
    const p = a.payload;
    const icon = `<div class="date-tile"><small>${esc(weekdayShort(new Date()))}</small><b>${dayOfMonth(new Date())}</b></div>`;
    if (a.status === "failed") return { icon, ...failed("read your calendar", a) };
    const items = (p.events || []).map((e) => `<li>${num(time(e.start))} ${esc(e.title)}</li>`).join("");
    return { icon, what: `Your calendar, ${esc(p.range)}`,
             how: items ? `<ul class="agenda">${items}</ul>` : "Nothing scheduled." };
  },
};

const STATUS_WORDS = { pending: "Working on it…", done: "Done", failed: "Failed",
                       cancelled: "Cancelled", triggered: "Happening now" };

function generic(a) {
  const icon = `<svg viewBox="-20 -20 40 40" aria-hidden="true"><circle r="14" fill="none" stroke="#C9A45C" stroke-width="1.6"/><circle r="3" fill="#C9A45C"/></svg>`;
  if (a.status === "failed") return { icon, what: esc(a.title || a.kind), how: esc(a.error || "Failed") };
  return { icon, what: esc(a.title || a.kind), how: STATUS_WORDS[a.status] || esc(a.status) };
}

const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);

export function cardHTML(action) {
  let parts;
  try {
    parts = (RENDERERS[action.kind] || generic)(action);
  } catch {
    parts = generic(action);   // malformed payload: still show something useful
  }
  return `<div class="icon">${parts.icon}</div><div class="body">` +
    `<div class="what">${parts.what}</div>` +
    (parts.how ? `<div class="how">${parts.how}</div>` : "") +
    (parts.extra ? `<div class="extra">${parts.extra}</div>` : "") + `</div>`;
}
