// Headless smoke test for web/app.js: a stub DOM, render every view and lesson, fail on exceptions.
//   node tools/smoke_web.js
const fs = require("fs");
const path = require("path");
const web = path.join(__dirname, "..", "web");

const els = new Map();
function el(sel) {
  if (!els.has(sel)) {
    const store = { innerHTML: "", textContent: "", hidden: false, value: "all", checked: false, dataset: {}, className: "" };
    store.classList = { toggle() {}, add() {}, remove() {} };
    store.addEventListener = () => {};
    store.querySelector = (s) => el(sel + " " + s);
    store.querySelectorAll = () => [];
    store.closest = () => null;
    store.showModal = () => {};
    els.set(sel, store);
  }
  return els.get(sel);
}
global.document = { querySelector: el, querySelectorAll: () => [], addEventListener: () => {}, createElement: () => el(Math.random()) };
const handlers = {};
global.window = { addEventListener: (ev, fn) => (handlers[ev] = fn), print: () => {} };
const mem = {};
global.localStorage = { getItem: (k) => mem[k] ?? null, setItem: (k, v) => (mem[k] = v) };
global.location = { hash: "", protocol: "file:" };
global.navigator = {};
global.fetch = async () => ({ json: async () => JSON.parse(fs.readFileSync(path.join(web, "course.json"), "utf8")) });

const errors = [];
process.on("unhandledRejection", (e) => errors.push(e));
eval(fs.readFileSync(path.join(web, "app.js"), "utf8"));

setTimeout(() => {
  const course = JSON.parse(fs.readFileSync(path.join(web, "course.json"), "utf8"));
  const main = el("#main");
  if (!main.innerHTML.includes("lesson")) errors.push(new Error("first lesson did not render"));
  const go = (hash, expect) => {
    location.hash = hash;
    try { handlers.hashchange(); } catch (e) { errors.push(new Error(`${hash}: ${e.stack}`)); return; }
    if (!main.innerHTML.includes(expect)) errors.push(new Error(`${hash}: expected "${expect}" in output`));
  };
  for (const l of course.lessons) go(`#lesson/${l.id}`, `${l.id}. `);
  go("#cards", "Card 1 of");
  go("#progress", "lab steps done");
  go("#lesson/06", "Measure");
  const st = JSON.parse(mem["gk-course-v1"] || "{}");
  console.log(`${course.lessons.length} lessons rendered; main ${main.innerHTML.length} chars; state keys: ${Object.keys(st).join(",")}`);
  if (errors.length) { console.error(errors); process.exit(1); }
  console.log("smoke ok");
}, 200);
