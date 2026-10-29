// Headless smoke test for web/app.js: a stub DOM, render every view and lesson, fail on exceptions.
//   node tools/smoke_web.js
const fs = require("fs");
const path = require("path");
const web = path.join(__dirname, "..", "web");

const els = new Map();
function el(sel) {
  if (!els.has(sel)) {
    const store = {
      innerHTML: "", textContent: "", hidden: false, value: "", checked: false, scrollTop: 0,
      dataset: {}, className: "", style: {}, files: [],
    };
    const classes = new Set();
    store.classList = {
      toggle(c) { classes.has(c) ? classes.delete(c) : classes.add(c); },
      add(c) { classes.add(c); },
      remove(c) { classes.delete(c); },
      contains(c) { return classes.has(c); },
    };
    store.addEventListener = (ev, fn) => { (store.handlers = store.handlers || {})[ev] = fn; };
    store.querySelector = (s) => el(sel + " " + s);
    store.querySelectorAll = () => [];
    store.closest = () => null;
    store.matches = () => false;
    store.focus = () => {};
    store.click = () => {};
    store.showModal = () => {};
    store.setAttribute = () => {};
    els.set(sel, store);
  }
  return els.get(sel);
}

const handlers = {};
global.document = {
  title: "",
  body: el("body"),
  querySelector: el,
  querySelectorAll: () => [],
  addEventListener: (ev, fn) => { (handlers[ev] = handlers[ev] || []).push(fn); },
  createElement: () => el("el-" + Math.random()),
};
global.window = {
  addEventListener: (ev, fn) => { handlers["w:" + ev] = fn; },
  removeEventListener: () => {},
  print: () => {},
  isSecureContext: false,
};
global.confirm = () => true;
global.alert = (m) => { throw new Error("alert: " + m); };
const mem = {};
global.localStorage = { getItem: (k) => (k in mem ? mem[k] : null), setItem: (k, v) => { mem[k] = v; } };
global.location = {
  _hash: "",
  protocol: "file:",
  get hash() { return this._hash; },
  set hash(v) {
    const next = v ? (v.startsWith("#") ? v : "#" + v) : "";
    if (next === this._hash) return;
    this._hash = next;
    if (handlers["w:hashchange"]) handlers["w:hashchange"]();
  },
};
global.navigator = {};
const course = JSON.parse(fs.readFileSync(path.join(web, "course.json"), "utf8"));
global.fetch = async () => ({ json: async () => JSON.parse(JSON.stringify(course)) });

const errors = [];
process.on("unhandledRejection", (e) => errors.push(e));
eval(fs.readFileSync(path.join(web, "app.js"), "utf8"));

setTimeout(() => {
  const stage = el("#stage");
  const actions = el("#actions");
  const status = el("#status");
  const drawerBody = el("#drawer-body");
  const check = (ok, msg) => { if (!ok) errors.push(new Error(msg)); };
  const fire = (type, target, extra = {}) => {
    try { for (const fn of handlers[type] || []) fn({ target, preventDefault() {}, ...extra }); }
    catch (e) { errors.push(e); }
  };
  const button = (dataset) => {
    const t = { dataset, disabled: false, matches: () => false };
    t.closest = (s) => (s === "[data-act]" ? t : null);
    return t;
  };
  const click = (dataset) => fire("click", button(dataset));
  const key = (k) => fire("keydown", { matches: () => false }, { key: k });
  const go = (hash, expect) => {
    try { location.hash = hash; } catch (e) { errors.push(new Error(`${hash}: ${e.stack}`)); return; }
    if (expect) check(stage.innerHTML.includes(expect), `${hash}: expected "${expect}"`);
  };

  check(stage.innerHTML.includes(`${course.lessons[0].id}. `), "first lesson did not render");
  for (const l of course.lessons) {
    go(`#lesson/${l.id}`, `${l.id}. `);
    check(status.innerHTML.includes(`Lesson ${l.id}`), `status missing for lesson ${l.id}`);
  }
  check(!/<pre><code class="lang-python">[^<]*\bdef\b/.test(stage.innerHTML), "code was not highlighted");

  // The drawer lists every lesson with a selection box and a progress ring.
  el("#open-drawer").handlers.click();
  check(el("#drawer").classList.contains("open"), "drawer did not open");
  check(course.lessons.every((l) => drawerBody.innerHTML.includes(`data-sel="${l.id}"`)), "drawer missing lessons");
  check(drawerBody.innerHTML.includes("Export progress") && drawerBody.innerHTML.includes("ring"), "drawer missing More panel or rings");
  el("#backdrop").handlers.click();
  check(!el("#drawer").classList.contains("open"), "backdrop did not close the drawer");

  // Lesson: steps, measurements, unlocking the explanation.
  const lab = course.lessons.find((l) => l.sections.some((s) => s.kind === "measure"));
  go(`#lesson/${lab.id}`, "Measure");
  check(stage.innerHTML.includes("Show the explanation anyway"), "explain was not locked before measuring");
  fire("change", { dataset: { step: "write0" }, checked: true });
  check(status.innerHTML.includes("1 of"), "step tick did not count");
  const fields = lab.sections.filter((s) => s.kind === "measure").flatMap((s) => s.fields);
  for (const f of fields) fire("change", { dataset: { key: f.key }, value: String(f.ref ?? 1) });
  check(!stage.innerHTML.includes("Show the explanation anyway"), "explain did not unlock after measuring");
  check(stage.innerHTML.includes("cmp good"), "measurement comparison missing");
  key("ArrowRight");
  check(location.hash !== `#lesson/${lab.id}`, "ArrowRight did not move to the next lesson");

  // Cards, Learn: mark known, move on.
  key("2");
  check(location.hash === "#cards" && stage.innerHTML.includes("flash"), "Cards view did not open");
  check(status.innerHTML.includes("of " + course.cards.length), "deck is not every card: " + status.innerHTML);
  click({ act: "known" });
  check(actions.innerHTML.includes("✓ Known"), "Known mark did not stick");
  click({ act: "card", delta: "1" });
  check(status.innerHTML.includes("Card 2"), "next card did not move");

  // Cards, Test: reveal, got, missed, finish, retest.
  click({ act: "mode", mode: "test" });
  check(actions.innerHTML.includes("Show answer") && stage.innerHTML.includes("hint"), "test card shows its answer before reveal");
  key(" ");
  check(actions.innerHTML.includes("Got it"), "Space did not reveal the answer");
  key("m");
  for (let i = 0; i < course.cards.length; i++) { key(" "); key("g"); }
  check(stage.innerHTML.includes("Retest the 1 missed"), "test summary missing: " + stage.innerHTML.slice(0, 200));
  click({ act: "retest" });
  check(status.innerHTML.includes("of 1"), "retest deck is not the one missed card");

  // Practise one lesson's cards from the lesson page.
  click({ act: "mode", mode: "learn" });
  go(`#lesson/${course.lessons[2].id}`);
  click({ act: "focus", id: course.lessons[2].id });
  const n = course.cards.filter((c) => c.lesson === course.lessons[2].id).length;
  check(!el("#focus").hidden && status.innerHTML.includes(`of ${n}`), "focused deck wrong: " + status.innerHTML);
  click({ act: "unfocus" });

  // Selection narrows the deck; clearing it empties the deck.
  fire("change", { dataset: { sel: course.lessons[0].id }, checked: false });
  check(!status.innerHTML.includes(`of ${course.cards.length}<`) , "deselecting a lesson did not shrink the deck");
  click({ act: "sel-none" });
  check(stage.innerHTML.includes("No cards match"), "empty selection did not empty the deck");
  click({ act: "sel-all" });

  go("#progress", "lab steps done");
  check(stage.innerHTML.includes("reference"), "progress missing measurement comparison");
  check(actions.innerHTML.includes("Continue"), "progress actions missing Continue");
  click({ act: "print-many" });
  check(el("body").classList.contains("print-many") && el("#print-area").innerHTML.includes("print-lesson"), "print selected lessons did not fill the print area");

  const st = JSON.parse(mem["gk-course-v1"] || "{}");
  check(st.cards && Object.keys(st.cards).length > 0 && st.measures[lab.id], "progress was not saved");
  if (errors.length) { for (const e of errors) console.error(e.stack || e); process.exit(1); }
  console.log(`smoke ok: ${course.lessons.length} lessons, ${course.cards.length} cards`);
  process.exit(0);
}, 50);
