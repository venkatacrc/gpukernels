/* GPU Kernels -> HSTU course app. Plain JavaScript, no build step, works offline. */
(() => {
  "use strict";

  const STORE = "gk-course-v1";
  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
  const main = $("#main");

  let course = null;
  let state = load();
  let deck = [];
  let deckPos = 0;
  let revealed = false;
  let score = { got: 0, missed: 0, missedIds: [] };

  function load() {
    const empty = { steps: {}, measures: {}, cards: {}, selected: null, view: "lessons", lesson: null,
                    cardMode: "learn", filter: "all", shuffle: false };
    try { return Object.assign(empty, JSON.parse(localStorage.getItem(STORE) || "{}")); }
    catch { return empty; }
  }
  function save() { localStorage.setItem(STORE, JSON.stringify(state)); }
  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => (t.hidden = true), 1800);
  }
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  /* ------------------------------------------------------------ progress model */
  function stepsOf(lesson) {
    return lesson.sections.filter((s) => ["write", "run", "measure", "explain"].includes(s.kind))
      .map((s, i) => `${s.kind}${i}`);
  }
  function measureFields(lesson) {
    return lesson.sections.filter((s) => s.kind === "measure").flatMap((s) => s.fields);
  }
  function lessonProgress(lesson) {
    const steps = stepsOf(lesson);
    const done = state.steps[lesson.id] || {};
    const n = steps.filter((k) => done[k]).length;
    return steps.length ? n / steps.length : 0;
  }
  function measuresDone(lesson) {
    const fields = measureFields(lesson);
    const m = state.measures[lesson.id] || {};
    return fields.length > 0 && fields.every((f) => m[f.key] !== undefined && m[f.key] !== "");
  }
  function selectedIds() {
    return state.selected || course.lessons.map((l) => l.id);
  }

  /* ------------------------------------------------------------ sidebar */
  function ring(frac) {
    const r = 9, c = 2 * Math.PI * r;
    return `<svg class="ring" viewBox="0 0 22 22" aria-label="${Math.round(frac * 100)}% done">
      <circle cx="11" cy="11" r="${r}" fill="none" stroke="var(--line)" stroke-width="3"/>
      <circle cx="11" cy="11" r="${r}" fill="none" stroke="var(--accent)" stroke-width="3"
        stroke-dasharray="${c * frac} ${c}" transform="rotate(-90 11 11)" stroke-linecap="round"/></svg>`;
  }
  function renderSidebar() {
    const sel = new Set(selectedIds());
    $("#lessonList").innerHTML = course.lessons.map((l) => `
      <li data-id="${l.id}" class="${state.view === "lessons" && state.lesson === l.id ? "on" : ""}">
        <input type="checkbox" ${sel.has(l.id) ? "checked" : ""} title="include in Cards and Print" aria-label="select ${esc(l.title)}">
        <span class="num">${l.id}</span>
        <span class="t">${esc(l.title)}<small>${l.minutes} min${l.track ? " · " + esc(l.track) : ""}</small></span>
        ${ring(lessonProgress(l))}
      </li>`).join("");
    const nCards = course.cards.filter((c) => sel.has(c.lesson)).length;
    $("#sideFoot").textContent = `${sel.size} lessons selected · ${nCards} cards · built ${course.built}`;
  }
  $("#lessonList").addEventListener("click", (e) => {
    const li = e.target.closest("li");
    if (!li) return;
    const id = li.dataset.id;
    if (e.target.matches("input")) {
      const sel = new Set(selectedIds());
      e.target.checked ? sel.add(id) : sel.delete(id);
      state.selected = course.lessons.map((l) => l.id).filter((x) => sel.has(x));
      save();
      if (state.view === "cards") buildDeck();
      renderSidebar();
      if (state.view === "cards") renderCards();
      return;
    }
    openLesson(id);
    $("#sidebar").classList.remove("open");
  });
  $("#selAll").onclick = () => { state.selected = null; save(); afterSelection(); };
  $("#selNone").onclick = () => { state.selected = []; save(); afterSelection(); };
  function afterSelection() { renderSidebar(); if (state.view === "cards") { buildDeck(); renderCards(); } }
  $("#menuBtn").onclick = () => $("#sidebar").classList.toggle("open");

  /* ------------------------------------------------------------ code blocks */
  const KEYWORDS = "def|return|for|in|if|else|elif|import|from|as|class|with|while|lambda|None|True|False|and|or|not|pass|raise|assert|yield|extern|__global__|__device__|__shared__|int|float|float4|const|void|long|export|source|exec";
  function highlight(text, lang) {
    // one left-to-right scan over the raw text, so a token can never land inside markup
    const comment = /^(cuda|cpp|c|cu)$/.test(lang) ? String.raw`\/\/[^\n]*` : String.raw`#[^\n]*`;
    const re = new RegExp(String.raw`(${comment})|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|(@[\w.]+)|\b(\d+(?:\.\d+)?(?:e-?\d+)?)\b|\b(${KEYWORDS})\b`, "g");
    const cls = ["tok-c", "tok-s", "tok-f", "tok-n", "tok-k"];
    let out = "", last = 0, m;
    while ((m = re.exec(text))) {
      const g = m.slice(1).findIndex((x) => x !== undefined);
      out += esc(text.slice(last, m.index)) + `<span class="${cls[g]}">${esc(m[0])}</span>`;
      last = m.index + m[0].length;
    }
    return out + esc(text.slice(last));
  }
  function decorateCode(root) {
    $$("pre > code", root).forEach((c) => {
      if (c.dataset.done) return;
      c.dataset.done = 1;
      const lang = (c.className.match(/lang-(\w+)/) || [])[1] || "python";
      c.innerHTML = highlight(c.textContent, lang);
      const btn = document.createElement("button");
      btn.className = "copy";
      btn.textContent = "copy";
      btn.onclick = () => navigator.clipboard.writeText(c.textContent).then(() => toast("Copied"));
      c.parentElement.appendChild(btn);
    });
  }

  /* ------------------------------------------------------------ lesson view */
  const LABEL = { write: "Write", run: "Run", measure: "Measure", explain: "Explain", note: "Note" };
  function compare(val, ref) {
    if (ref === null || ref === undefined || val === "" || isNaN(+val)) return "";
    const r = +val / ref;
    const cls = r > 0.8 && r < 1.25 ? "good" : r > 0.5 && r < 2 ? "warn" : "bad";
    return `<span class="cmp ${cls}">reference ${fmt(ref)} · yours is ${r >= 1 ? r.toFixed(2) + "×" : (r * 100).toFixed(0) + "%"}</span>`;
  }
  const fmt = (x) => (Math.abs(x) >= 100 ? Math.round(x).toLocaleString() : (+x).toPrecision(3));

  function lessonHTML(lesson, forPrint = false) {
    const done = state.steps[lesson.id] || {};
    const m = state.measures[lesson.id] || {};
    const unlocked = forPrint || measuresDone(lesson) || done.revealed || !measureFields(lesson).length;
    let stepIdx = 0;
    const parts = lesson.sections.map((s) => {
      if (s.kind === "read") return `<div class="read">${s.html}</div>`;
      const key = ["write", "run", "measure", "explain"].includes(s.kind) ? `${s.kind}${stepIdx++}` : null;
      const isDone = key && done[key];
      const tick = key && !forPrint ? `<label class="done"><input type="checkbox" data-step="${key}" ${isDone ? "checked" : ""}> done</label>` : "";
      let body;
      if (s.kind === "measure") {
        body = `<p>Run the lab, then record what <em>your</em> GPU printed. The reference value from the course GB200 appears after you save.</p>
          <table class="mtable">${s.fields.map((f) => `
            <tr><td>${esc(f.label)}</td>
              <td><input inputmode="decimal" data-key="${f.key}" value="${esc(m[f.key] ?? "")}" ${forPrint ? "disabled" : ""}> <span class="unit">${esc(f.unit)}</span></td>
              <td>${m[f.key] !== undefined && m[f.key] !== "" ? compare(m[f.key], f.ref) : forPrint && f.ref !== null ? `<span class="cmp">reference ${fmt(f.ref)}</span>` : ""}</td></tr>`).join("")}
          </table>`;
      } else if (s.kind === "explain" && !unlocked) {
        body = `<div class="locked">Record your measurements first: predicting and then checking is the point of the lab.<br>
          <button class="btn" data-reveal>Show the explanation anyway</button></div>`;
      } else {
        body = s.html;
      }
      return `<section class="step ${s.kind} ${isDone ? "is-done" : ""}">
        <header><span class="badge">${LABEL[s.kind]}</span>${tick}</header>
        <div class="body">${body}</div></section>`;
    }).join("");
    const lab = lesson.lab ? ` · lab <code>${esc(lesson.lab)}</code>` : "";
    return `<article class="lesson ${forPrint ? "print-lesson" : ""}" data-id="${lesson.id}">
      <div class="lesson-head"><h1>${lesson.id}. ${esc(lesson.title)}</h1></div>
      <div class="meta">${lesson.minutes} min${lab}${lesson.summary ? " · " + esc(lesson.summary) : ""}</div>
      ${parts}</article>`;
  }

  function openLesson(id) {
    state.view = "lessons";
    state.lesson = id;
    save();
    location.hash = `#lesson/${id}`;
    render();
  }

  function renderLesson() {
    const idx = Math.max(0, course.lessons.findIndex((l) => l.id === state.lesson));
    const lesson = course.lessons[idx];
    state.lesson = lesson.id;
    const prev = course.lessons[idx - 1], next = course.lessons[idx + 1];
    main.innerHTML = lessonHTML(lesson) + `<div class="pager">
      ${prev ? `<button class="btn" data-go="${prev.id}">← ${esc(prev.title)}</button>` : "<span></span>"}
      ${next ? `<button class="btn primary" data-go="${next.id}">${esc(next.title)} →</button>` : ""}</div>`;
    decorateCode(main);
    main.scrollTop = 0;

    $$("[data-go]", main).forEach((b) => (b.onclick = () => openLesson(b.dataset.go)));
    $$("[data-step]", main).forEach((cb) => (cb.onchange = () => {
      state.steps[lesson.id] = Object.assign(state.steps[lesson.id] || {}, { [cb.dataset.step]: cb.checked });
      save();
      cb.closest(".step").classList.toggle("is-done", cb.checked);
      renderSidebar();
    }));
    $$("[data-key]", main).forEach((inp) => (inp.onchange = () => {
      const v = inp.value.trim().replace(/,/g, "");
      state.measures[lesson.id] = Object.assign(state.measures[lesson.id] || {}, { [inp.dataset.key]: v });
      if (measuresDone(lesson)) {
        const mIdx = stepsOf(lesson).find((k) => k.startsWith("measure"));
        if (mIdx) state.steps[lesson.id] = Object.assign(state.steps[lesson.id] || {}, { [mIdx]: true });
      }
      save();
      const y = main.scrollTop;
      renderLesson();
      main.scrollTop = y;
      renderSidebar();
    }));
    const rev = $("[data-reveal]", main);
    if (rev) rev.onclick = () => {
      state.steps[lesson.id] = Object.assign(state.steps[lesson.id] || {}, { revealed: true });
      save();
      const y = main.scrollTop;
      renderLesson();
      main.scrollTop = y;
    };
  }

  /* ------------------------------------------------------------ cards view */
  function buildDeck() {
    const sel = new Set(selectedIds());
    deck = course.cards.filter((c) => sel.has(c.lesson)).filter((c) => {
      const s = state.cards[c.id];
      if (state.filter === "unknown") return s !== "known";
      if (state.filter === "review") return s === "review";
      return true;
    });
    if (state.shuffle) {
      for (let i = deck.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [deck[i], deck[j]] = [deck[j], deck[i]];
      }
    }
    deckPos = 0;
    revealed = false;
    score = { got: 0, missed: 0, missedIds: [] };
  }
  function lessonTitle(id) { const l = course.lessons.find((x) => x.id === id); return l ? `${l.id}. ${l.title}` : id; }

  function renderCards() {
    const test = state.cardMode === "test";
    if (!deck.length) {
      main.innerHTML = `<div class="empty">No cards match. Select lessons in the sidebar or change the filter.</div>`;
      return;
    }
    if (deckPos >= deck.length) {
      const total = score.got + score.missed;
      main.innerHTML = `<div class="cardview"><div class="flash"><div class="q">${test ? `Score: ${score.got} / ${total}` : "End of deck"}</div>
        <div class="a">${test && score.missed ? `${score.missed} missed — they are now marked for review.` : "Nice work."}</div>
        <div class="card-actions">${test && score.missed ? `<button class="btn primary" id="retest">Retest missed cards</button>` : ""}
        <button class="btn" id="again">Start over</button></div></div></div>`;
      const rt = $("#retest");
      if (rt) rt.onclick = () => {
        const ids = new Set(score.missedIds);
        deck = course.cards.filter((c) => ids.has(c.id));
        deckPos = 0; revealed = false; score = { got: 0, missed: 0, missedIds: [] };
        renderCards();
      };
      $("#again").onclick = () => { buildDeck(); renderCards(); };
      return;
    }
    const c = deck[deckPos];
    const mark = state.cards[c.id];
    const showA = !test || revealed;
    main.innerHTML = `<div class="cardview">
      <div class="scorebar"><span>Card ${deckPos + 1} of ${deck.length}</span>
        ${test ? `<span>Got ${score.got} · Missed ${score.missed}</span>` : ""}</div>
      <div class="flash" id="flash">
        <div class="from">${esc(lessonTitle(c.lesson))}</div>
        ${mark ? `<span class="tag ${mark}">${mark}</span>` : ""}
        <div class="q">${c.q}</div>
        ${showA ? `<div class="a">${c.a}</div>` : `<div class="hint">Recall the answer, then press <kbd>Space</kbd> or tap the card</div>`}
      </div>
      <div class="card-actions">
        <button class="btn" data-act="prev">← Prev</button>
        ${test && !revealed ? `<button class="btn primary" data-act="reveal">Show answer</button>` : ""}
        ${test && revealed ? `<button class="btn primary" data-act="got">Got it (G)</button><button class="btn" data-act="missed">Missed it (M)</button>` : ""}
        ${!test ? `<button class="btn" data-act="known">Known (K)</button><button class="btn" data-act="review">Review (R)</button>` : ""}
        <button class="btn" data-act="next">${test ? "Skip" : "Next"} →</button>
      </div></div>`;
    $$("[data-act]", main).forEach((b) => (b.onclick = () => cardAct(b.dataset.act)));
    const flash = $("#flash");
    flash.onclick = (e) => { if (test && !revealed && !e.target.closest("a")) cardAct("reveal"); };
    let x0 = null;
    flash.addEventListener("touchstart", (e) => (x0 = e.touches[0].clientX), { passive: true });
    flash.addEventListener("touchend", (e) => {
      if (x0 === null) return;
      const dx = e.changedTouches[0].clientX - x0;
      if (Math.abs(dx) > 60) cardAct(dx < 0 ? "next" : "prev");
      x0 = null;
    });
  }
  function cardAct(act) {
    const c = deck[deckPos];
    if (act === "prev") { deckPos = Math.max(0, deckPos - 1); revealed = false; }
    else if (act === "next") { deckPos++; revealed = false; }
    else if (act === "reveal") { revealed = true; }
    else if (act === "known" || act === "review") {
      state.cards[c.id] = state.cards[c.id] === act ? undefined : act;
      save();
    } else if (act === "got" || act === "missed") {
      if (!revealed) return;
      if (act === "got") { score.got++; state.cards[c.id] = "known"; }
      else { score.missed++; score.missedIds.push(c.id); state.cards[c.id] = "review"; }
      save();
      deckPos++;
      revealed = false;
    }
    renderCards();
  }

  /* ------------------------------------------------------------ progress view */
  function renderProgress() {
    const ls = course.lessons;
    const stepsTotal = ls.reduce((a, l) => a + stepsOf(l).length, 0);
    const stepsDone = ls.reduce((a, l) => a + stepsOf(l).filter((k) => (state.steps[l.id] || {})[k]).length, 0);
    const known = course.cards.filter((c) => state.cards[c.id] === "known").length;
    const review = course.cards.filter((c) => state.cards[c.id] === "review").length;
    const mTotal = ls.reduce((a, l) => a + measureFields(l).length, 0);
    const mDone = ls.reduce((a, l) => a + measureFields(l).filter((f) => ((state.measures[l.id] || {})[f.key] ?? "") !== "").length, 0);
    main.innerHTML = `<h1>Progress</h1>
      <div class="kpis">
        <div class="kpi"><b>${stepsDone}/${stepsTotal}</b><span>lab steps done</span></div>
        <div class="kpi"><b>${mDone}/${mTotal}</b><span>measurements recorded</span></div>
        <div class="kpi"><b>${known}</b><span>cards known</span></div>
        <div class="kpi"><b>${review}</b><span>cards to review</span></div>
      </div>
      <table class="ptable"><thead><tr><th>Lesson</th><th>Steps</th><th></th><th>Your measurements vs reference</th></tr></thead><tbody>
      ${ls.map((l) => {
        const fr = lessonProgress(l);
        const m = state.measures[l.id] || {};
        const cmp = measureFields(l).filter((f) => (m[f.key] ?? "") !== "").map((f) => `${esc(f.label)}: <b>${esc(m[f.key])}</b> ${compare(m[f.key], f.ref)}`).join("<br>");
        return `<tr><td><a href="#lesson/${l.id}">${l.id}. ${esc(l.title)}</a></td>
          <td class="n">${stepsOf(l).filter((k) => (state.steps[l.id] || {})[k]).length}/${stepsOf(l).length}</td>
          <td><div class="bar"><i style="width:${fr * 100}%"></i></div></td><td>${cmp || '<span class="meta">—</span>'}</td></tr>`;
      }).join("")}</tbody></table>
      <p class="meta">Progress is stored in this browser only. Use ⋮ → Export progress to move it to another device.</p>`;
  }

  /* ------------------------------------------------------------ views */
  function render() {
    $$(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.view === state.view));
    $(".card-controls").hidden = state.view !== "cards";
    $$("#cardMode button").forEach((b) => b.classList.toggle("on", b.dataset.mode === state.cardMode));
    $("#cardFilter").value = state.filter;
    $("#shuffle").checked = state.shuffle;
    renderSidebar();
    if (state.view === "lessons") renderLesson();
    else if (state.view === "cards") renderCards();
    else renderProgress();
  }
  $$(".tabs button").forEach((b) => (b.onclick = () => setView(b.dataset.view)));
  function setView(v) {
    state.view = v;
    save();
    if (v === "cards") buildDeck();
    location.hash = v === "lessons" ? `#lesson/${state.lesson}` : `#${v}`;
    render();
  }
  $$("#cardMode button").forEach((b) => (b.onclick = () => { state.cardMode = b.dataset.mode; save(); buildDeck(); render(); }));
  $("#cardFilter").onchange = (e) => { state.filter = e.target.value; save(); buildDeck(); render(); };
  $("#shuffle").onchange = (e) => { state.shuffle = e.target.checked; save(); buildDeck(); render(); };
  $("#restart").onclick = () => { buildDeck(); renderCards(); };

  /* ------------------------------------------------------------ print */
  function doPrint() {
    if (state.view === "lessons") { window.print(); return; }
    const sel = new Set(selectedIds());
    const html = course.lessons.filter((l) => sel.has(l.id)).map((l) => lessonHTML(l, true)).join("");
    const saved = main.innerHTML;
    main.innerHTML = html;
    decorateCode(main);
    window.print();
    main.innerHTML = saved;
    render();
  }
  $("#printBtn").onclick = doPrint;

  /* ------------------------------------------------------------ menu */
  $("#moreBtn").onclick = (e) => { e.stopPropagation(); $("#moreMenu").hidden = !$("#moreMenu").hidden; };
  document.addEventListener("click", () => ($("#moreMenu").hidden = true));
  $("#exportBtn").onclick = () => {
    const blob = new Blob([JSON.stringify({ app: "gk-course", version: 1, exported: new Date().toISOString(), state }, null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `gk-progress-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
  };
  $("#importBtn").onclick = () => $("#importFile").click();
  $("#importFile").onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try {
      const data = JSON.parse(await f.text());
      if (data.app !== "gk-course") throw new Error("not a course progress file");
      state = Object.assign(load(), data.state);
      save();
      render();
      toast("Progress imported");
    } catch (err) { alert("Import failed: " + err.message); }
    e.target.value = "";
  };
  $("#resetBtn").onclick = () => {
    if (!confirm("Clear all steps, measurements and card marks in this browser?")) return;
    state = Object.assign(load(), { steps: {}, measures: {}, cards: {} });
    save();
    render();
  };
  $("#helpBtn").onclick = () => $("#helpDlg").showModal();

  /* ------------------------------------------------------------ keyboard */
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea, select")) return;
    if ((e.metaKey || e.ctrlKey) && e.key === "p") { e.preventDefault(); doPrint(); return; }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "1") return setView("lessons");
    if (e.key === "2") return setView("cards");
    if (e.key === "3") return setView("progress");
    if (state.view === "cards") {
      const k = e.key.toLowerCase();
      const map = { arrowleft: "prev", arrowright: "next", " ": state.cardMode === "test" ? "reveal" : null,
                    g: "got", m: "missed", k: "known", r: "review" };
      if (map[k]) { e.preventDefault(); cardAct(map[k]); }
    } else if (state.view === "lessons" && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
      const i = course.lessons.findIndex((l) => l.id === state.lesson) + (e.key === "ArrowRight" ? 1 : -1);
      if (course.lessons[i]) openLesson(course.lessons[i].id);
    }
  });

  /* ------------------------------------------------------------ routing + boot */
  function fromHash() {
    const h = location.hash.slice(1);
    if (h.startsWith("lesson/")) { state.view = "lessons"; state.lesson = h.slice(7); }
    else if (h === "cards" || h === "progress") state.view = h;
  }
  window.addEventListener("hashchange", () => {
    const before = `${state.view}/${state.lesson}`;
    fromHash();
    if (`${state.view}/${state.lesson}` !== before) { if (state.view === "cards") buildDeck(); render(); }
  });

  fetch("course.json", { cache: "no-cache" })
    .then((r) => r.json())
    .then((c) => {
      course = c;
      if (!state.lesson || !course.lessons.some((l) => l.id === state.lesson)) state.lesson = course.lessons[0].id;
      fromHash();
      if (state.view === "cards") buildDeck();
      render();
    })
    .catch((err) => (main.innerHTML = `<div class="empty">Could not load course.json (${esc(err.message)}).<br>
      Serve the folder over HTTP: <code>python -m http.server -d web 8080</code></div>`));

  if ("serviceWorker" in navigator && location.protocol !== "file:") {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  }
})();
