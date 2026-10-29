// Renders lessons inside the Mac app's web view. The app passes window.GK =
// { lessons, steps, measures, print } and listens on webkit.messageHandlers.gk.
(() => {
    'use strict';

    const GK = window.GK;
    const STEP_KINDS = ['write', 'run', 'measure', 'explain'];
    const LABEL = { write: 'Write', run: 'Run', measure: 'Measure', explain: 'Explain', note: 'Note' };
    const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    const unesc = (s) => s.replace(/&(lt|gt|quot|#39|amp);/g, (_, e) => ({ lt: '<', gt: '>', quot: '"', '#39': "'", amp: '&' }[e]));
    const post = (msg) => {
        if (window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.gk) {
            window.webkit.messageHandlers.gk.postMessage(msg);
        }
    };

    function stepsOf(lesson) {
        let i = 0;
        return lesson.sections.filter((s) => STEP_KINDS.includes(s.kind)).map((s) => `${s.kind}${i++}`);
    }
    function measureFields(lesson) {
        return lesson.sections.filter((s) => s.kind === 'measure').flatMap((s) => s.fields || []);
    }
    function measuresDone(lesson) {
        const fields = measureFields(lesson);
        const m = GK.measures[lesson.id] || {};
        return fields.length > 0 && fields.every((f) => m[f.key] !== undefined && m[f.key] !== '');
    }
    const fmt = (x) => (Math.abs(x) >= 100 ? Math.round(x).toLocaleString() : (+x).toPrecision(3));
    function compare(val, ref) {
        if (ref === null || ref === undefined || val === '' || val === undefined || isNaN(+val)) return '';
        const r = +val / ref;
        const cls = r > 0.8 && r < 1.25 ? 'good' : r > 0.5 && r < 2 ? 'warn' : 'bad';
        return `<span class="cmp ${cls}">reference ${fmt(ref)} · yours is ${r >= 1 ? r.toFixed(2) + '×' : (r * 100).toFixed(0) + '%'}</span>`;
    }

    const KEYWORDS = 'def|return|for|in|if|else|elif|import|from|as|class|with|while|lambda|None|True|False|and|or|not|pass|raise|assert|yield|extern|__global__|__device__|__shared__|int|float|float4|const|void|long|export|source|exec';
    function highlight(text, lang) {
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
    function decorate(html) {
        return html.replace(/<pre><code class="lang-(\w+)">([\s\S]*?)<\/code><\/pre>/g, (_, lang, body) => {
            const text = unesc(body);
            const code = lang === 'text' ? esc(text) : highlight(text, lang);
            return `<pre>${GK.print ? '' : '<button type="button" class="copy">Copy</button>'}<code class="lang-${lang}">${code}</code></pre>`;
        });
    }

    function measureHTML(lesson, s) {
        const m = GK.measures[lesson.id] || {};
        return `<p>Run the lab, then record what <em>your</em> GPU printed. The reference value from the course GB200 appears after you enter yours.</p>
            ${(s.fields || []).map((f) => {
                const has = m[f.key] !== undefined && m[f.key] !== '';
                const cmp = has ? compare(m[f.key], f.ref) : GK.print && f.ref != null ? `<span class="cmp">reference ${fmt(f.ref)}</span>` : '';
                return `<div class="measure-field">
                    <label for="m-${lesson.id}-${f.key}">${esc(f.label)}</label>
                    <div class="mval"><input id="m-${lesson.id}-${f.key}" inputmode="decimal" autocomplete="off" data-lesson="${lesson.id}" data-key="${esc(f.key)}" value="${esc(m[f.key] ?? '')}" ${GK.print ? 'disabled' : ''}><span class="unit">${esc(f.unit)}</span></div>
                    <div class="cmp-line">${cmp}</div>
                </div>`;
            }).join('')}`;
    }

    function lessonHTML(lesson) {
        const done = GK.steps[lesson.id] || {};
        const unlocked = GK.print || done.revealed || measuresDone(lesson) || !measureFields(lesson).length;
        let stepIdx = 0;
        const parts = lesson.sections.map((s) => {
            if (s.kind === 'read') return `<div class="prose">${decorate(s.html || '')}</div>`;
            const key = STEP_KINDS.includes(s.kind) ? `${s.kind}${stepIdx++}` : null;
            const isDone = Boolean(key && done[key]);
            const tick = key && !GK.print
                ? `<label class="done"><input type="checkbox" data-lesson="${lesson.id}" data-step="${key}" ${isDone ? 'checked' : ''}> Done</label>` : '';
            let body;
            if (s.kind === 'measure') body = measureHTML(lesson, s);
            else if (s.kind === 'explain' && !unlocked) {
                body = `<div class="locked">Record your measurements first: predicting and then checking is the point of the lab.<br>
                    <button type="button" class="btn" data-reveal="${lesson.id}">Show the explanation anyway</button></div>`;
            } else body = decorate(s.html || '');
            return `<section class="step ${s.kind}${isDone ? ' is-done' : ''}">
                <header><span class="badge">${LABEL[s.kind] || s.kind}</span>${tick}</header>
                <div class="body">${body}</div></section>`;
        }).join('');
        const summary = lesson.summary ? lesson.summary[0].toUpperCase() + lesson.summary.slice(1) + '.' : '';
        const meta = [`${lesson.minutes} min`];
        if (lesson.lab) meta.push(`lab <code>${esc(lesson.lab)}</code>`);
        return `<article class="lesson${GK.print ? ' print-lesson' : ''}">
            ${lesson.track ? `<p class="kicker">${esc(lesson.track)}</p>` : ''}
            <h1>${lesson.id}. ${esc(lesson.title)}</h1>
            <p class="meta">${meta.join(' · ')}</p>
            ${summary ? `<p class="lead">${esc(summary)}</p>` : ''}
            ${parts}</article>`;
    }

    const root = document.getElementById('root');
    function render() {
        const y = window.scrollY;
        root.innerHTML = GK.lessons.map(lessonHTML).join('');
        window.scrollTo(0, y);
    }
    const lessonById = (id) => GK.lessons.find((l) => l.id === id);

    document.addEventListener('change', (e) => {
        const t = e.target;
        const lesson = lessonById(t.dataset.lesson);
        if (!lesson) return;
        if (t.dataset.step) {
            GK.steps[lesson.id] = Object.assign(GK.steps[lesson.id] || {}, { [t.dataset.step]: t.checked });
            t.closest('.step').classList.toggle('is-done', t.checked);
            post({ type: 'step', lesson: lesson.id, key: t.dataset.step, on: t.checked });
        } else if (t.dataset.key) {
            const wasUnlocked = measuresDone(lesson);
            const value = t.value.trim().replace(/,/g, '');
            GK.measures[lesson.id] = Object.assign(GK.measures[lesson.id] || {}, { [t.dataset.key]: value });
            post({ type: 'measure', lesson: lesson.id, key: t.dataset.key, value });
            const f = measureFields(lesson).find((x) => x.key === t.dataset.key);
            const line = t.closest('.measure-field').querySelector('.cmp-line');
            if (line && f) line.innerHTML = value === '' ? '' : compare(value, f.ref);
            if (!wasUnlocked && measuresDone(lesson)) {
                const k = stepsOf(lesson).find((s) => s.startsWith('measure'));
                if (k) GK.steps[lesson.id] = Object.assign(GK.steps[lesson.id] || {}, { [k]: true });
                render();
            }
        }
    });

    document.addEventListener('click', (e) => {
        const reveal = e.target.closest('[data-reveal]');
        if (reveal) {
            const id = reveal.dataset.reveal;
            GK.steps[id] = Object.assign(GK.steps[id] || {}, { revealed: true });
            post({ type: 'reveal', lesson: id });
            render();
            return;
        }
        const copy = e.target.closest('.copy');
        if (copy) {
            const code = copy.parentElement.querySelector('code');
            post({ type: 'copy', text: code ? code.textContent : '' });
            copy.textContent = 'Copied';
            setTimeout(() => { copy.textContent = 'Copy'; }, 1200);
        }
    });

    render();
})();
