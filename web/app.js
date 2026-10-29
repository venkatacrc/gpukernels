// GPU Kernels -> HSTU course app. Plain JavaScript, no build step, works offline.
// course.json comes from tools/build_course.py.
(() => {
    'use strict';

    const STORE = 'gk-course-v1';
    const VIEWS = [['lessons', 'Lessons'], ['cards', 'Cards'], ['progress', 'Progress']];
    const MODES = [['learn', 'Learn'], ['test', 'Test']];
    const STEP_KINDS = ['write', 'run', 'measure', 'explain'];
    const LABEL = { write: 'Write', run: 'Run', measure: 'Measure', explain: 'Explain', note: 'Note' };
    const $ = (sel) => document.querySelector(sel);
    const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    const unesc = (s) => s.replace(/&(lt|gt|quot|#39|amp);/g, (_, e) => ({ lt: '<', gt: '>', quot: '"', '#39': "'", amp: '&' }[e]));

    let course = null;
    let state = load();
    let deck = [];
    let pos = 0;
    let revealed = false;
    let finished = false;
    let score = { got: 0, missed: 0, missedIds: [] };

    function blank() {
        return {
            steps: {}, measures: {}, cards: {}, selected: null, view: 'lessons', lesson: null,
            cardMode: 'learn', filter: 'all', shuffle: false, focus: null,
        };
    }
    function load() {
        try { return Object.assign(blank(), JSON.parse(localStorage.getItem(STORE) || '{}')); }
        catch { return blank(); }
    }
    function save() { localStorage.setItem(STORE, JSON.stringify(state)); }

    function toast(msg) {
        const t = $('#toast');
        t.textContent = msg;
        t.hidden = false;
        clearTimeout(toast.timer);
        toast.timer = setTimeout(() => { t.hidden = true; }, 2000);
    }

    // ---- progress model ----

    function stepsOf(lesson) {
        let i = 0;
        return lesson.sections.filter((s) => STEP_KINDS.includes(s.kind)).map((s) => `${s.kind}${i++}`);
    }
    function measureFields(lesson) {
        return lesson.sections.filter((s) => s.kind === 'measure').flatMap((s) => s.fields);
    }
    function stepsDone(lesson) {
        const done = state.steps[lesson.id] || {};
        return stepsOf(lesson).filter((k) => done[k]).length;
    }
    function lessonProgress(lesson) {
        const n = stepsOf(lesson).length;
        return n ? stepsDone(lesson) / n : 0;
    }
    function measuresDone(lesson) {
        const fields = measureFields(lesson);
        const m = state.measures[lesson.id] || {};
        return fields.length > 0 && fields.every((f) => m[f.key] !== undefined && m[f.key] !== '');
    }
    function selectedIds() { return state.selected || course.lessons.map((l) => l.id); }
    function lessonById(id) { return course.lessons.find((l) => l.id === id); }
    function currentLesson() { return lessonById(state.lesson) || course.lessons[0]; }
    function lessonTitle(id) { const l = lessonById(id); return l ? `${l.id}. ${l.title}` : id; }
    function cardsOf(id) { return course.cards.filter((c) => c.lesson === id); }

    function compare(val, ref) {
        if (ref === null || ref === undefined || val === '' || val === undefined || isNaN(+val)) return '';
        const r = +val / ref;
        const cls = r > 0.8 && r < 1.25 ? 'good' : r > 0.5 && r < 2 ? 'warn' : 'bad';
        return `<span class="cmp ${cls}">reference ${fmt(ref)} · yours is ${r >= 1 ? r.toFixed(2) + '×' : (r * 100).toFixed(0) + '%'}</span>`;
    }
    const fmt = (x) => (Math.abs(x) >= 100 ? Math.round(x).toLocaleString() : (+x).toPrecision(3));

    // ---- code blocks ----

    const KEYWORDS = 'def|return|for|in|if|else|elif|import|from|as|class|with|while|lambda|None|True|False|and|or|not|pass|raise|assert|yield|extern|__global__|__device__|__shared__|int|float|float4|const|void|long|export|source|exec';
    function highlight(text, lang) {
        // One left-to-right scan over the raw text, so a token can never land inside markup.
        const comment = /^(cuda|cpp|c|cu)$/.test(lang) ? String.raw`\/\/[^\n]*` : String.raw`#[^\n]*`;
        const re = new RegExp(String.raw`(${comment})|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|(@[\w.]+)|\b(\d+(?:\.\d+)?(?:e-?\d+)?)\b|\b(${KEYWORDS})\b`, 'g');
        const cls = ['tok-c', 'tok-s', 'tok-f', 'tok-n', 'tok-k'];
        let out = '';
        let last = 0;
        let m;
        while ((m = re.exec(text))) {
            const g = m.slice(1).findIndex((x) => x !== undefined);
            out += esc(text.slice(last, m.index)) + `<span class="${cls[g]}">${esc(m[0])}</span>`;
            last = m.index + m[0].length;
        }
        return out + esc(text.slice(last));
    }
    const decorated = new Map();
    function decorate(html, copy = true) {
        const key = (copy ? 'c' : 'p') + html;
        if (!decorated.has(key)) {
            decorated.set(key, html.replace(/<pre><code class="lang-(\w+)">([\s\S]*?)<\/code><\/pre>/g, (_, lang, body) => {
                const text = unesc(body);
                const code = lang === 'text' ? esc(text) : highlight(text, lang);
                return `<pre>${copy ? '<button type="button" class="copy" data-act="copy">Copy</button>' : ''}<code class="lang-${lang}">${code}</code></pre>`;
            }));
        }
        return decorated.get(key);
    }

    // ---- lesson ----

    function measureHTML(lesson, s, forPrint) {
        const m = state.measures[lesson.id] || {};
        return `<p>Run the lab, then record what <em>your</em> GPU printed. The reference value from the course GB200 appears after you enter yours.</p>
            ${s.fields.map((f) => {
                const has = m[f.key] !== undefined && m[f.key] !== '';
                const cmp = has ? compare(m[f.key], f.ref) : forPrint && f.ref != null ? `<span class="cmp">reference ${fmt(f.ref)}</span>` : '';
                return `<div class="measure-field">
                    <label for="m-${lesson.id}-${f.key}">${esc(f.label)}</label>
                    <div class="mval"><input id="m-${lesson.id}-${f.key}" inputmode="decimal" autocomplete="off" data-key="${esc(f.key)}" value="${esc(m[f.key] ?? '')}" ${forPrint ? 'disabled' : ''}><span class="unit">${esc(f.unit)}</span></div>
                    <div class="cmp-line">${cmp}</div>
                </div>`;
            }).join('')}`;
    }

    function lessonHTML(lesson, forPrint = false) {
        const done = state.steps[lesson.id] || {};
        const unlocked = forPrint || done.revealed || measuresDone(lesson) || !measureFields(lesson).length;
        let stepIdx = 0;
        const parts = lesson.sections.map((s) => {
            if (s.kind === 'read') return `<div class="prose">${decorate(s.html, !forPrint)}</div>`;
            const key = STEP_KINDS.includes(s.kind) ? `${s.kind}${stepIdx++}` : null;
            const isDone = Boolean(key && done[key]);
            const tick = key && !forPrint
                ? `<label class="done"><input type="checkbox" data-step="${key}" ${isDone ? 'checked' : ''}> Done</label>` : '';
            let body;
            if (s.kind === 'measure') body = measureHTML(lesson, s, forPrint);
            else if (s.kind === 'explain' && !unlocked) {
                body = `<div class="locked">Record your measurements first: predicting and then checking is the point of the lab.<br>
                    <button type="button" class="btn" data-act="reveal-explain">Show the explanation anyway</button></div>`;
            } else body = decorate(s.html, !forPrint);
            return `<section class="step ${s.kind}${isDone ? ' is-done' : ''}">
                <header><span class="badge">${LABEL[s.kind] || s.kind}</span>${tick}</header>
                <div class="body">${body}</div></section>`;
        }).join('');
        const summary = lesson.summary ? lesson.summary[0].toUpperCase() + lesson.summary.slice(1) + '.' : '';
        const meta = [`${lesson.minutes} min`];
        if (lesson.lab) meta.push(`lab <code>${esc(lesson.lab)}</code>`);
        return `${lesson.track ? `<p class="kicker">${esc(lesson.track)}</p>` : ''}
            <h1>${lesson.id}. ${esc(lesson.title)}</h1>
            <p class="meta">${meta.join(' · ')}</p>
            ${summary ? `<p class="lead">${esc(summary)}</p>` : ''}
            ${parts}`;
    }

    function openLesson(id) {
        if (!lessonById(id)) return;
        state.view = 'lessons';
        state.lesson = id;
        save();
        const hash = `#lesson/${id}`;
        if (location.hash !== hash) location.hash = hash;
        else render();
        const card = $('#card');
        if (card) card.scrollTop = 0;
    }
    function stepLesson(delta) {
        const i = course.lessons.indexOf(currentLesson()) + delta;
        if (course.lessons[i]) openLesson(course.lessons[i].id);
    }

    // ---- cards ----

    function buildDeck() {
        const ids = state.focus && lessonById(state.focus) ? new Set([state.focus]) : new Set(selectedIds());
        deck = course.cards.filter((c) => ids.has(c.lesson)).filter((c) => {
            const s = state.cards[c.id];
            if (state.filter === 'unknown') return s !== 'known';
            if (state.filter === 'review') return s === 'review';
            return true;
        });
        if (state.shuffle) {
            for (let i = deck.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [deck[i], deck[j]] = [deck[j], deck[i]];
            }
        }
        pos = 0;
        revealed = false;
        finished = false;
        score = { got: 0, missed: 0, missedIds: [] };
    }
    function moveCard(delta) {
        if (!deck.length || finished) return;
        const next = pos + delta;
        if (next < 0) return;
        if (next >= deck.length) {
            if (state.cardMode === 'test') finished = true;
            else return;
        } else pos = next;
        revealed = false;
        render();
    }
    function answer(got) {
        if (state.cardMode !== 'test' || !revealed || finished) return;
        const c = deck[pos];
        if (got) { score.got++; state.cards[c.id] = 'known'; }
        else { score.missed++; score.missedIds.push(c.id); state.cards[c.id] = 'review'; }
        save();
        if (pos + 1 >= deck.length) finished = true;
        else pos++;
        revealed = false;
        render();
    }
    function mark(kind) {
        const c = deck[pos];
        if (!c || state.cardMode !== 'learn') return;
        if (state.cards[c.id] === kind) delete state.cards[c.id];
        else state.cards[c.id] = kind;
        save();
        render();
    }
    function reveal() {
        if (state.cardMode !== 'test' || revealed || finished || !deck.length) return;
        revealed = true;
        render();
    }

    // ---- views ----

    function setView(v) {
        state.view = v;
        save();
        if (v === 'cards') buildDeck();
        const hash = v === 'lessons' ? `#lesson/${currentLesson().id}` : `#${v}`;
        if (location.hash !== hash) location.hash = hash;
        else render();
    }

    function render() {
        renderSwitches();
        renderFocus();
        renderStatus();
        renderStage();
        renderActions();
        if ($('#drawer').classList.contains('open')) renderDrawer();
        document.title = state.view === 'lessons' ? `${lessonTitle(currentLesson().id)} · GPU Kernels → HSTU` : 'GPU Kernels → HSTU';
    }

    function renderSwitches() {
        $('#view-switch').innerHTML = VIEWS.map(([v, t]) =>
            `<button type="button" data-act="view" data-view="${v}" class="${state.view === v ? 'on' : ''}">${t}</button>`).join('');
        const modes = $('#mode-switch');
        modes.hidden = state.view !== 'cards';
        modes.innerHTML = MODES.map(([v, t]) =>
            `<button type="button" data-act="mode" data-mode="${v}" class="${state.cardMode === v ? 'on' : ''}">${t}</button>`).join('');
    }

    function renderFocus() {
        const el = $('#focus');
        const on = state.view === 'cards' && state.focus && lessonById(state.focus);
        el.hidden = !on;
        el.innerHTML = on
            ? `<span>Cards from <strong>${esc(lessonTitle(state.focus))}</strong></span>
               <button type="button" data-act="unfocus">All selected lessons</button>` : '';
    }

    function renderStatus() {
        let left = '';
        let right = '';
        if (state.view === 'lessons') {
            const l = currentLesson();
            left = `<strong>Lesson ${l.id}</strong> · ${l.minutes} min${l.track ? ' · ' + esc(l.track) : ''}`;
            right = `${stepsDone(l)} of ${stepsOf(l).length} steps`;
        } else if (state.view === 'cards') {
            if (deck.length && !finished) {
                left = `<strong>Card ${pos + 1}</strong> of ${deck.length}`;
                if (state.cardMode === 'test') {
                    const answered = score.got + score.missed;
                    right = `Score ${score.got} of ${answered} · ${deck.length - answered} to go`;
                } else {
                    const known = deck.filter((c) => state.cards[c.id] === 'known').length;
                    const review = deck.filter((c) => state.cards[c.id] === 'review').length;
                    right = `Known ${known} · Review ${review}`;
                }
            }
        } else {
            const total = course.lessons.reduce((a, l) => a + stepsOf(l).length, 0);
            const done = course.lessons.reduce((a, l) => a + stepsDone(l), 0);
            left = '<strong>Progress</strong>';
            right = `${total ? Math.round((done / total) * 100) : 0}% of lab steps`;
        }
        $('#status').innerHTML = `<span>${left}</span><span>${right}</span>`;
    }

    function renderStage() {
        let html;
        if (state.view === 'lessons') html = `<article class="card lesson" id="card">${lessonHTML(currentLesson())}</article>`;
        else if (state.view === 'cards') html = cardHTML();
        else html = `<article class="card" id="card">${progressHTML()}</article>`;
        $('#stage').innerHTML = html;
    }

    function cardHTML() {
        if (!deck.length) {
            return `<div class="card empty" id="card">
                <p>No cards match. Select lessons in the ☰ menu, or show all cards instead of a filtered set.</p>
                <button type="button" class="btn primary" data-act="drawer">Choose lessons</button></div>`;
        }
        const test = state.cardMode === 'test';
        if (finished) {
            const total = score.got + score.missed;
            const pct = deck.length ? Math.round((score.got / deck.length) * 100) : 0;
            return `<div class="card flash summary" id="card">
                <div class="big">${score.got} / ${total}</div>
                <p>${pct}% of ${deck.length} cards. ${score.missed ? `The ${score.missed} you missed are marked for review.` : 'Every card answered is marked known.'}</p>
                <div class="stack">
                    ${score.missed ? `<button type="button" class="btn primary" data-act="retest">Retest the ${score.missed} missed</button>` : ''}
                    <button type="button" class="btn" data-act="restart">Start a new test</button>
                    <button type="button" class="btn" data-act="mode" data-mode="learn">Switch to Learn</button>
                </div></div>`;
        }
        const c = deck[pos];
        const m = state.cards[c.id];
        const showA = !test || revealed;
        return `<article class="card flash${test && revealed ? ' flipped' : ''}" id="card" ${test && !revealed ? 'data-act="reveal"' : ''}>
            <div class="badges"><span class="label">${esc(lessonTitle(c.lesson))}</span>${m ? `<span class="tag ${m}">${m === 'known' ? 'Known' : 'Review'}</span>` : ''}</div>
            <div class="q">${c.q}</div>
            ${showA ? `<div class="a">${c.a}</div>` : '<div class="hint">Recall the answer, then tap the card or press Space.</div>'}
        </article>`;
    }

    function ring(frac) {
        const r = 9;
        const c = 2 * Math.PI * r;
        return `<svg class="ring" viewBox="0 0 22 22" aria-label="${Math.round(frac * 100)}% done">
            <circle cx="11" cy="11" r="${r}" fill="none" stroke="var(--line)" stroke-width="3"/>
            <circle cx="11" cy="11" r="${r}" fill="none" stroke="var(--accent-light)" stroke-width="3"
                stroke-dasharray="${c * frac} ${c}" transform="rotate(-90 11 11)" stroke-linecap="round"/></svg>`;
    }

    function progressHTML() {
        const ls = course.lessons;
        const total = ls.reduce((a, l) => a + stepsOf(l).length, 0);
        const done = ls.reduce((a, l) => a + stepsDone(l), 0);
        const known = course.cards.filter((c) => state.cards[c.id] === 'known').length;
        const review = course.cards.filter((c) => state.cards[c.id] === 'review').length;
        const mTotal = ls.reduce((a, l) => a + measureFields(l).length, 0);
        const mDone = ls.reduce((a, l) => a + measureFields(l).filter((f) => ((state.measures[l.id] || {})[f.key] ?? '') !== '').length, 0);
        const rows = ls.map((l) => {
            const m = state.measures[l.id] || {};
            const lines = measureFields(l).filter((f) => (m[f.key] ?? '') !== '')
                .map((f) => `${esc(f.label)}: <b>${esc(m[f.key])}</b> ${esc(f.unit)} ${compare(m[f.key], f.ref)}`);
            return `<button type="button" class="row" data-act="open" data-id="${l.id}">
                    <span class="num">${l.id}</span>
                    <span class="grow">${esc(l.title)}<small>${stepsDone(l)} of ${stepsOf(l).length} steps</small></span>
                    <span class="bar"><i style="width:${lessonProgress(l) * 100}%"></i></span>
                </button>${lines.length ? `<div class="measures-list">${lines.join('<br>')}</div>` : ''}`;
        }).join('');
        return `<p class="kicker">${esc(course.title)}</p>
            <h1>Progress</h1>
            <div class="kpis">
                <div class="kpi"><b>${done}/${total}</b><span>lab steps done</span></div>
                <div class="kpi"><b>${mDone}/${mTotal}</b><span>measurements</span></div>
                <div class="kpi"><b>${known}</b><span>cards known</span></div>
                <div class="kpi"><b>${review}</b><span>cards to review</span></div>
            </div>
            <h4>Lessons</h4>
            <div class="panel">${rows}</div>
            <p class="fine">Progress stays in this browser. Export it from the ☰ menu to move it to another device or to the Mac app.</p>
            <p class="fine install-hint">On a Pixel: Chrome ⋮ menu → <strong>Install app</strong> (or <strong>Add to Home screen</strong>). It keeps working offline.</p>`;
    }

    function renderActions() {
        const a = $('#actions');
        const prev = (act, off) => `<button type="button" class="btn nav" data-act="${act}" data-delta="-1" aria-label="Previous" ${off ? 'disabled' : ''}>‹</button>`;
        const next = (act, off, label = 'Next') => `<button type="button" class="btn nav" data-act="${act}" data-delta="1" aria-label="${label}" ${off ? 'disabled' : ''}>›</button>`;
        if (state.view === 'lessons') {
            const l = currentLesson();
            const i = course.lessons.indexOf(l);
            const n = cardsOf(l.id).length;
            a.innerHTML = prev('lesson', i <= 0)
                + `<button type="button" class="btn" data-act="focus" data-id="${l.id}" ${n ? '' : 'disabled'}>Practise ${n} cards</button>`
                + next('lesson', i >= course.lessons.length - 1);
            return;
        }
        if (state.view === 'progress') {
            const nextUp = course.lessons.find((l) => lessonProgress(l) < 1) || course.lessons[0];
            a.innerHTML = `<button type="button" class="btn" data-act="export">Export progress</button>
                <button type="button" class="btn primary" data-act="open" data-id="${nextUp.id}">Continue: lesson ${nextUp.id} ›</button>`;
            return;
        }
        if (!deck.length || finished) { a.innerHTML = ''; return; }
        const c = deck[pos];
        if (state.cardMode === 'learn') {
            const m = state.cards[c.id];
            a.innerHTML = prev('card', pos <= 0)
                + `<button type="button" class="btn${m === 'known' ? ' on-known' : ''}" data-act="known">${m === 'known' ? '✓ Known' : 'Known'}</button>`
                + `<button type="button" class="btn${m === 'review' ? ' on-review' : ''}" data-act="review">${m === 'review' ? '⚑ Review' : 'Review'}</button>`
                + next('card', pos >= deck.length - 1);
        } else if (!revealed) {
            a.innerHTML = prev('card', pos <= 0)
                + '<button type="button" class="btn primary" data-act="reveal">Show answer</button>'
                + next('card', false, 'Skip');
        } else {
            a.innerHTML = prev('card', pos <= 0)
                + '<button type="button" class="btn bad" data-act="missed">✗ Missed it</button>'
                + '<button type="button" class="btn good" data-act="got">✓ Got it</button>'
                + next('card', false, 'Skip');
        }
    }

    // ---- drawer ----

    function openDrawer() {
        renderDrawer();
        $('#drawer').classList.add('open');
        $('#drawer').setAttribute('aria-hidden', 'false');
        $('#backdrop').hidden = false;
    }
    function closeDrawer() {
        $('#drawer').classList.remove('open');
        $('#drawer').setAttribute('aria-hidden', 'true');
        $('#backdrop').hidden = true;
    }

    function renderDrawer() {
        const body = $('#drawer-body');
        const scroll = body.scrollTop;
        const sel = new Set(selectedIds());
        const rows = course.lessons.map((l) => {
            const on = state.view === 'lessons' && currentLesson().id === l.id;
            return `<div class="row${on ? ' on' : ''}">
                <input type="checkbox" data-sel="${l.id}" ${sel.has(l.id) ? 'checked' : ''} aria-label="Include lesson ${l.id} in Cards and Print">
                <button type="button" class="open" data-act="open" data-id="${l.id}">
                    <span class="num">${l.id}</span>
                    <span class="grow">${esc(l.title)}<small>${l.minutes} min${l.track ? ' · ' + esc(l.track) : ''} · ${cardsOf(l.id).length} cards</small></span>
                </button>
                ${ring(lessonProgress(l))}
            </div>`;
        }).join('');
        const filters = [['all', 'All cards'], ['unknown', 'Not yet known'], ['review', 'Review only']]
            .map(([v, t]) => `<option value="${v}" ${state.filter === v ? 'selected' : ''}>${t}</option>`).join('');
        body.innerHTML = `<h4>Lessons</h4>
            <div class="panel">
                <div class="links"><button type="button" data-act="sel-all">Select all</button><button type="button" data-act="sel-none">Clear</button></div>
                ${rows}
            </div>
            <h4>Cards</h4>
            <div class="panel">
                <div class="row"><span class="grow">Practise</span><select id="card-filter" aria-label="Which cards">${filters}</select></div>
                <div class="row"><input type="checkbox" id="shuffle" ${state.shuffle ? 'checked' : ''}><label for="shuffle">Shuffle</label></div>
                <button type="button" class="menu-btn" data-act="restart">↺ Restart from the first card</button>
            </div>
            <h4>More</h4>
            <div class="panel">
                <button type="button" class="menu-btn" data-act="print-lesson">🖨 Print this lesson</button>
                <button type="button" class="menu-btn" data-act="print-many" ${sel.size ? '' : 'disabled'}>🖨 Print the ${sel.size} selected lessons</button>
                <button type="button" class="menu-btn" data-act="export">⤓ Export progress</button>
                <button type="button" class="menu-btn" data-act="import">⤒ Import progress</button>
                <button type="button" class="menu-btn" data-act="help">⌨ Keyboard shortcuts</button>
                <a class="menu-btn" href="https://github.com/venkatacrc/gpukernels" target="_blank" rel="noopener">↗ Repository</a>
                <button type="button" class="menu-btn danger" data-act="reset">Reset progress</button>
            </div>`;
        body.scrollTop = scroll;
        const nCards = course.cards.filter((c) => sel.has(c.lesson)).length;
        $('#drawer-foot').textContent = `${sel.size} lessons selected · ${nCards} cards`;
    }

    function setSelected(ids) {
        const all = course.lessons.map((l) => l.id);
        state.selected = ids.length === all.length ? null : all.filter((x) => ids.includes(x));
        save();
        if (state.view === 'cards') buildDeck();
        render();
    }

    // ---- print ----

    function printLesson() {
        if (state.view !== 'lessons') { openLesson(currentLesson().id); }
        closeDrawer();
        setTimeout(() => window.print(), 250);
    }
    function printSelected() {
        const sel = new Set(selectedIds());
        const area = $('#print-area');
        area.innerHTML = course.lessons.filter((l) => sel.has(l.id))
            .map((l) => `<article class="card print-lesson">${lessonHTML(l, true)}</article>`).join('');
        document.body.classList.add('print-many');
        closeDrawer();
        const done = () => {
            document.body.classList.remove('print-many');
            area.innerHTML = '';
            window.removeEventListener('afterprint', done);
        };
        window.addEventListener('afterprint', done);
        setTimeout(() => window.print(), 250);
    }
    function printAny() {
        if (state.view === 'lessons') printLesson();
        else printSelected();
    }

    // ---- progress in and out ----

    function exportProgress() {
        const data = { app: 'gk-course', version: 1, exported: new Date().toISOString(), state };
        const blob = new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `gk-progress-${new Date().toISOString().slice(0, 10)}.json`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    }
    function importProgress(file) {
        const reader = new FileReader();
        reader.onload = () => {
            try {
                const data = JSON.parse(reader.result);
                if (data.app !== 'gk-course') throw new Error('not a course progress file');
                state = Object.assign(blank(), data.state);
                save();
                if (state.view === 'cards') buildDeck();
                render();
                toast('Progress imported');
            } catch (err) { alert('Import failed: ' + err.message); }
        };
        reader.readAsText(file);
    }

    function copyCode(btn) {
        const code = btn.closest && btn.closest('pre') && btn.closest('pre').querySelector('code');
        const text = code ? code.textContent : '';
        if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(() => toast('Copied'), () => toast('Copy failed'));
        else toast('Select the text to copy it');
    }

    // ---- events ----

    function keepScroll(fn) {
        const card = $('#card');
        const y = card ? card.scrollTop : 0;
        fn();
        const again = $('#card');
        if (again) again.scrollTop = y;
    }

    function act(el) {
        const d = el.dataset;
        switch (d.act) {
            case 'view': setView(d.view); break;
            case 'mode':
                state.cardMode = d.mode;
                save();
                buildDeck();
                render();
                break;
            case 'open': closeDrawer(); openLesson(d.id); break;
            case 'lesson': stepLesson(Number(d.delta)); break;
            case 'card': moveCard(Number(d.delta)); break;
            case 'focus':
                state.focus = d.id;
                setView('cards');
                break;
            case 'unfocus':
                state.focus = null;
                save();
                buildDeck();
                render();
                break;
            case 'reveal': reveal(); break;
            case 'got': answer(true); break;
            case 'missed': answer(false); break;
            case 'known': mark('known'); break;
            case 'review': mark('review'); break;
            case 'retest': {
                const ids = new Set(score.missedIds);
                deck = course.cards.filter((c) => ids.has(c.id));
                pos = 0;
                revealed = false;
                finished = false;
                score = { got: 0, missed: 0, missedIds: [] };
                render();
                break;
            }
            case 'restart': buildDeck(); closeDrawer(); render(); break;
            case 'reveal-explain': {
                const l = currentLesson();
                state.steps[l.id] = Object.assign(state.steps[l.id] || {}, { revealed: true });
                save();
                keepScroll(render);
                break;
            }
            case 'drawer': openDrawer(); break;
            case 'sel-all': setSelected(course.lessons.map((l) => l.id)); break;
            case 'sel-none': setSelected([]); break;
            case 'copy': copyCode(el); break;
            case 'print-lesson': printLesson(); break;
            case 'print-many': printSelected(); break;
            case 'export': exportProgress(); break;
            case 'import': $('#import-file').click(); break;
            case 'help': closeDrawer(); $('#help-dialog').showModal(); break;
            case 'reset':
                if (!confirm('Clear all steps, measurements and card marks in this browser?')) return;
                state = Object.assign(blank(), { view: state.view, lesson: state.lesson, cardMode: state.cardMode });
                save();
                buildDeck();
                render();
                toast('Progress cleared');
                break;
            default: break;
        }
    }

    document.addEventListener('click', (e) => {
        const el = e.target.closest ? e.target.closest('[data-act]') : null;
        if (!el || el.disabled) return;
        if (el.id === 'card' && e.target.closest && e.target.closest('a')) return;
        act(el);
    });

    document.addEventListener('change', (e) => {
        const t = e.target;
        if (t.dataset && t.dataset.step) {
            const l = currentLesson();
            state.steps[l.id] = Object.assign(state.steps[l.id] || {}, { [t.dataset.step]: t.checked });
            save();
            keepScroll(render);
        } else if (t.dataset && t.dataset.key) {
            const l = currentLesson();
            const wasUnlocked = measuresDone(l);
            const v = t.value.trim().replace(/,/g, '');
            state.measures[l.id] = Object.assign(state.measures[l.id] || {}, { [t.dataset.key]: v });
            if (measuresDone(l)) {
                const k = stepsOf(l).find((s) => s.startsWith('measure'));
                if (k) state.steps[l.id] = Object.assign(state.steps[l.id] || {}, { [k]: true });
            }
            save();
            const field = t.closest && t.closest('.measure-field');
            const line = field && field.querySelector('.cmp-line');
            const f = measureFields(l).find((x) => x.key === t.dataset.key);
            if (line && f) line.innerHTML = v === '' ? '' : compare(v, f.ref);
            // Re-render only when the explanation unlocks, so focus can move to the next field.
            if (!wasUnlocked && measuresDone(l)) keepScroll(render);
            else { renderStatus(); if ($('#drawer').classList.contains('open')) renderDrawer(); }
        } else if (t.dataset && t.dataset.sel) {
            const set = new Set(selectedIds());
            t.checked ? set.add(t.dataset.sel) : set.delete(t.dataset.sel);
            setSelected([...set]);
        } else if (t.id === 'card-filter') {
            state.filter = t.value;
            save();
            buildDeck();
            render();
        } else if (t.id === 'shuffle') {
            state.shuffle = t.checked;
            save();
            buildDeck();
            render();
        } else if (t.id === 'import-file' && t.files && t.files[0]) {
            importProgress(t.files[0]);
            t.value = '';
        }
    });

    document.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'p') { e.preventDefault(); printAny(); return; }
        if (e.key === 'Escape' && $('#drawer').classList.contains('open')) { closeDrawer(); return; }
        if (e.target.matches && e.target.matches('input, textarea, select')) return;
        if (e.metaKey || e.ctrlKey || e.altKey) return;
        if (e.key === '1' || e.key === '2' || e.key === '3') { setView(VIEWS[Number(e.key) - 1][0]); return; }
        const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
        if (state.view === 'cards') {
            const map = {
                ArrowLeft: () => moveCard(-1), ArrowRight: () => moveCard(1), ' ': reveal,
                g: () => answer(true), m: () => answer(false), k: () => mark('known'), r: () => mark('review'),
            };
            if (map[k]) { e.preventDefault(); map[k](); }
        } else if (state.view === 'lessons' && (k === 'ArrowLeft' || k === 'ArrowRight')) {
            stepLesson(k === 'ArrowRight' ? 1 : -1);
        }
    });

    $('#open-drawer').addEventListener('click', openDrawer);
    $('#close-drawer').addEventListener('click', closeDrawer);
    $('#backdrop').addEventListener('click', closeDrawer);
    $('#open-print').addEventListener('click', printAny);

    // Swipe the lesson or the card left or right, as in Sloka Words.
    let x0 = null;
    let y0 = null;
    const stage = $('#stage');
    stage.addEventListener('touchstart', (e) => {
        if (e.target.closest && e.target.closest('pre, table, input, select')) { x0 = null; return; }
        x0 = e.touches[0].clientX;
        y0 = e.touches[0].clientY;
    }, { passive: true });
    stage.addEventListener('touchend', (e) => {
        if (x0 == null) return;
        const dx = e.changedTouches[0].clientX - x0;
        const dy = e.changedTouches[0].clientY - y0;
        x0 = null;
        if (Math.abs(dx) < 60 || Math.abs(dx) < Math.abs(dy) * 1.5) return;
        if (state.view === 'cards') moveCard(dx < 0 ? 1 : -1);
        else if (state.view === 'lessons') stepLesson(dx < 0 ? 1 : -1);
    }, { passive: true });

    // ---- routing and boot ----

    function fromHash() {
        const h = decodeURIComponent(location.hash.slice(1));
        if (h.startsWith('lesson/') && lessonById(h.slice(7))) { state.view = 'lessons'; state.lesson = h.slice(7); }
        else if (h === 'cards' || h === 'progress') state.view = h;
    }
    window.addEventListener('hashchange', () => {
        const before = state.view;
        fromHash();
        if (state.view === 'cards' && before !== 'cards') buildDeck();
        save();
        render();
    });

    fetch('course.json', { cache: 'no-cache' })
        .then((r) => r.json())
        .then((c) => {
            course = c;
            if (!lessonById(state.lesson)) state.lesson = course.lessons[0].id;
            fromHash();
            buildDeck();
            render();
        })
        .catch((err) => {
            $('#stage').innerHTML = `<div class="card empty">Could not load course.json (${esc(err.message)}).<br>
                Serve the folder over HTTP: <code>python -m http.server -d web 8080</code></div>`;
        });

    if ('serviceWorker' in navigator && location.protocol.startsWith('http')) {
        navigator.serviceWorker.register('sw.js').catch(() => {});
    }
})();
