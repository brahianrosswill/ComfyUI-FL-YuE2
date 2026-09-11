export function el(tag, text, parent) {
    const item = document.createElement(tag);
    if (text) item.textContent = text;
    parent?.append(item);
    return item;
}

export function button(parent, text, action, secondary = false) {
    const item = el("button", text, parent);
    item.type = "button"; item.className = secondary ? "secondary" : ""; item.onclick = action;
    return item;
}

function combo(parent, title, options, value, change) {
    const item = el("select", null, el("label", title, parent));
    for (const option of options) {
        const [value, label] = Array.isArray(option) ? option : [option, option];
        el("option", label, item).value = value;
    }
    item.value = value; item.onchange = () => change(item.value);
    return item;
}

const PPQ = 256, KEYS = 48, ROW = 16, TOP = 24;
const pitchName = pitch => ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"][pitch % 12] + (Math.floor(pitch / 12) - 1);
function shape(tag, attributes, parent) {
    const item = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attributes)) item.setAttribute(key, value);
    parent.append(item); return item;
}

const pitchClass = name => ({C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11}[name[0]] + [...name.slice(1)].reduce((sum, accidental) => sum + (accidental === "#" ? 1 : accidental === "b" ? -1 : 0), 0) + 12) % 12;
export function transposeScore(data, key) {
    let delta = (pitchClass(key.replace(/m$/, "")) - pitchClass(data.key.replace(/m$/, "")) + 12) % 12;
    if (delta > 6) delta -= 12;
    const notes = Object.values(data.roll.tracks).flat();
    if (notes.some(note => note.pitch + delta < 0 || note.pitch + delta > 127)) throw new Error("That key would move a note outside the supported pitch range.");
    for (const note of notes) note.pitch += delta;
    const names = key.includes("b") || ["F", "Dm", "Gm", "Cm", "Fm"].includes(key) ? ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"] : ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    for (const chord of data.roll.chords) chord.symbol = chord.symbol.replace(/(^|\/)([A-G](?:bb|##|b|#)?)/g, (_, prefix, root) => prefix + names[(pitchClass(root) + delta + 12) % 12]);
    data.key = key;
}

export class PianoRoll {
    constructor(editor) { this.editor = editor; this.section = 0; this.track = "Ins"; this.snap = 64; this.low = 48; this.selected = new Set(); this.position = 0; this.playing = false; this.metronome = false; }
    get data() { return this.editor.data; }
    row(parent) { const item = el("div", null, parent); item.className = "row"; return item; }
    bounds() {
        const section = this.data.roll.sections[this.section], start = section.start * this.data.roll.bar_ticks;
        return {section, start, end: start + section.bars * this.data.roll.bar_ticks};
    }
    render() {
        const scroll = {left: this.scroll?.scrollLeft || 0, top: this.scroll?.scrollTop || 0};
        this.section = Math.min(this.section, this.data.roll.sections.length - 1); this.selected.clear();
        const root = this.editor.piano; root.replaceChildren();
        const controls = this.row(root);
        const tempo = el("input", null, el("label", "Tempo", controls));
        tempo.type = "number"; tempo.min = 1; tempo.max = 1000; tempo.value = this.data.bpm;
        tempo.onchange = () => this.editor.commit(data => { data.bpm = Number(tempo.value); });
        const keyInput = combo(controls, "Key", this.data.keys, this.data.key, key => this.editor.commit(data => transposeScore(data, key)));
        keyInput.title = "Transpose all notes and chord roots by the same interval. Rhythm and melodic intervals are preserved; changing major/minor does not rewrite the melody.";
        combo(controls, "Melody", [["Ins", "Instrumental"], ["Vocal", "Sung melody"]], this.track, track => { this.pause(); this.track = track; this.render(); });
        combo(controls, "Snap", [[256, "Quarter notes"], [128, "Eighth notes"], [64, "Sixteenth notes"], [32, "32nd notes"]], this.snap, snap => { this.snap = Number(snap); this.draw(); });
        button(controls, "Center notes", () => this.fit(), true);
        const sections = this.row(root);
        combo(sections, "Section", this.data.roll.sections.map((s, i) => [i, `${i + 1}. ${s.name}`]), this.section, index => { this.pause(); this.section = Number(index); this.position = this.bounds().start; this.render(); });
        const arrangement = el("details", null, root); el("summary", "Arrange sections", arrangement);
        const structure = this.row(arrangement);
        const name = el("input", null, structure); name.value = this.bounds().section.name; name.style.width = "120px";
        name.setAttribute("aria-label", "Section name"); name.onchange = () => this.editor.commit(data => { data.roll.sections[this.section].name = name.value; });
        button(structure, "+ Section", () => this.editor.commit(data => { data.roll.sections.push({name: "new section", bars: 4}); }));
        button(structure, "Duplicate", () => this.duplicateSection(), true);
        button(structure, "Earlier", () => this.moveSection(-1), true).disabled = this.section === 0;
        button(structure, "Later", () => this.moveSection(1), true).disabled = this.section === this.data.roll.sections.length - 1;
        button(structure, "Remove section", () => this.removeSection(), true).disabled = this.data.roll.sections.length === 1;
        const tools = this.row(root);
        this.playButton = button(tools, this.playing ? "Pause" : "Play", () => this.togglePlayback()); button(tools, "Stop", () => this.stop(), true);
        const metronome = el("input", null, el("label", "Metronome", tools)); metronome.type = "checkbox"; metronome.checked = this.metronome;
        metronome.onchange = () => { const playing = this.playing; this.pause(); this.metronome = metronome.checked; if (playing) this.play(); this.editor.root.focus(); };
        button(tools, "Delete selected", () => this.deleteNote(), true);
        button(tools, "Clear melody", () => this.editor.commit(data => { data.roll.tracks[this.track] = []; }), true);
        button(tools, "+ Bar", () => this.addBar(), true);
        button(tools, "Octave ↓", () => { this.low = Math.max(0, this.low - 12); this.draw(); }, true);
        button(tools, "Octave ↑", () => { this.low = Math.min(92, this.low + 12); this.draw(); }, true);
        button(tools, "Instrumental only", () => this.editor.commit(data => { data.roll.tracks.Vocal = []; this.track = "Ins"; }), true);
        el("p", "Shift-drag to box-select notes; Shift-click toggles a note. Drag selected notes to move them together. Draw on empty space; drag a single note’s right edge to resize. Drag the timeline to scrub. Space plays/pauses; arrows move the selection; Shift + up/down shifts an octave. Delete removes; Ctrl/Cmd + Z undoes.", root).className = "help";
        this.noteStatus = el("div", "Select a note to see pitch, position and length.", root); this.noteStatus.className = "help";
        this.scroll = el("div", null, root); this.scroll.className = "roll";
        this.draw(); this.scroll.scrollLeft = scroll.left; this.scroll.scrollTop = scroll.top; this.renderChords();
    }
    draw() {
        this.scroll.replaceChildren(); const {start, end, section} = this.bounds();
        this.zoom = ((this.scroll.clientWidth || 760) - KEYS) * PPQ / (end - start);
        const PX = this.zoom;
        const width = KEYS + (end - start) / PPQ * PX, height = TOP + 36 * ROW;
        this.svg = shape("svg", {width, height, viewBox: `0 0 ${width} ${height}`, role: "application", "aria-label": `${this.track} piano roll`}, this.scroll);
        for (let row = 0; row < 36; row++) {
            const pitch = this.low + 35 - row, black = [1, 3, 6, 8, 10].includes(pitch % 12);
            shape("rect", {x: 0, y: TOP + row * ROW, width, height: ROW, fill: black ? "#211029" : "#302038", stroke: "#44304e", "stroke-width": 0.5}, this.svg);
        }
        for (let tick = 0; tick <= end - start; tick += this.snap) {
            const x = KEYS + tick / PPQ * PX;
            shape("line", {x1: x, x2: x, y1: TOP, y2: height, stroke: tick % this.data.roll.bar_ticks === 0 ? "#b293c0" : "#573460", "stroke-width": 0.5}, this.svg);
        }
        shape("rect", {x: KEYS, y: 0, width: width - KEYS, height: TOP, fill: "#16727c", cursor: "ew-resize"}, this.svg);
        const labelEvery = Math.max(1, Math.ceil(60 / (this.data.roll.bar_ticks / PPQ * PX)));
        for (let bar = 0; bar < section.bars; bar += labelEvery) shape("text", {x: KEYS + bar * this.data.roll.bar_ticks / PPQ * PX + 5, y: 16, fill: "#fff", "font-size": 11, "pointer-events": "none"}, this.svg).textContent = `Bar ${section.start + bar + 1}`;
        this.noteRects = [];
        this.data.roll.tracks[this.track].forEach((note, index) => {
            if (note.start >= end || note.start + note.duration <= start || note.pitch < this.low || note.pitch >= this.low + 36) return;
            const x = KEYS + (Math.max(start, note.start) - start) / PPQ * PX;
            const w = (Math.min(end, note.start + note.duration) - Math.max(start, note.start)) / PPQ * PX;
            const rect = shape("rect", {x, y: TOP + (this.low + 35 - note.pitch) * ROW + 1, width: w, height: ROW - 2, rx: 3,
                fill: this.track === "Ins" ? "#43cfb1" : "#c28aff", stroke: "#16727c", "stroke-width": 1, "data-note": index, cursor: "grab"}, this.svg);
            this.noteRects.push({index, rect});
            shape("title", {}, rect).textContent = `${pitchName(note.pitch)} · ${note.duration / PPQ} beats`;
            if (w >= 8) shape("rect", {x: x + w - 4, y: TOP + (this.low + 35 - note.pitch) * ROW + 3, width: 3, height: ROW - 6, fill: "#fff", opacity: 0.65, "data-note": index, cursor: "ew-resize"}, this.svg);
            if (w >= 32) shape("text", {x: x + 3, y: TOP + (this.low + 35 - note.pitch) * ROW + 12, fill: "#18352f", "font-size": 10, "pointer-events": "none"}, this.svg).textContent = pitchName(note.pitch);
        });
        const keyboard = shape("g", {}, this.svg);
        shape("rect", {x: 0, y: 0, width: KEYS, height: TOP, fill: "#29003d"}, keyboard);
        for (let row = 0; row < 36; row++) {
            const pitch = this.low + 35 - row, black = [1, 3, 6, 8, 10].includes(pitch % 12);
            shape("rect", {x: 0, y: TOP + row * ROW, width: KEYS, height: ROW, fill: black ? "#392342" : "#e6dfea", stroke: "#765287", cursor: "pointer"}, keyboard);
            shape("text", {x: 4, y: TOP + row * ROW + 12, fill: black ? "#ddd" : "#211029", "font-size": 10, "pointer-events": "none"}, keyboard).textContent = pitchName(pitch);
        }
        this.scroll.onscroll = () => keyboard.setAttribute("transform", `translate(${this.scroll.scrollLeft || 0},0)`);
        this.scroll.onscroll();
        this.cursor = shape("line", {y1: 0, y2: height, stroke: "#fff", "stroke-width": 2, cursor: "ew-resize", "data-playhead": "true"}, this.svg);
        this.cursorHandle = shape("rect", {y: 1, width: 10, height: TOP - 2, rx: 3, fill: "#fff", cursor: "ew-resize", "data-playhead": "true"}, this.svg);
        this.updateCursor();
        this.showSelection();
        this.svg.oncontextmenu = event => { event.preventDefault(); const index = event.target.getAttribute("data-note"); if (index !== null) { this.selected = new Set([Number(index)]); this.deleteNote(); } };
        this.svg.onpointerdown = event => this.pointer(event);
    }
    pointer(event) {
        const PX = this.zoom;
        if (event.button !== 0 || this.editor.busy) return;
        event.preventDefault(); event.stopPropagation(); this.editor.root.focus({preventScroll: true});
        const {start, end} = this.bounds(), svg = this.svg, rect = svg.getBoundingClientRect();
        const width = Number(svg.getAttribute("width")), height = Number(svg.getAttribute("height"));
        const point = e => ({x: (e.clientX - rect.left) * width / rect.width, y: (e.clientY - rect.top) * height / rect.height});
        const first = point(event);
        if (event.shiftKey && first.x >= KEYS && first.y >= TOP) { this.boxSelect(event, point); return; }
        if ((first.y < TOP && first.x >= KEYS) || event.target.getAttribute("data-playhead")) { this.scrub(event, point); return; }
        if (first.y < TOP) return;
        if (first.x < KEYS + (this.scroll.scrollLeft || 0)) { this.audition(this.low + 35 - Math.floor((first.y - TOP) / ROW)); return; }
        const hit = event.target.getAttribute("data-note"), index = hit === null ? -1 : Number(hit);
        this.pause();
        if (this.selected.has(index) && this.selected.size > 1) { this.moveSelection(event, point, index); return; }
        const old = index < 0 ? null : structuredClone(this.data.roll.tracks[this.track][index]); this.selected = new Set(index < 0 ? [] : [index]);
        this.showSelection();
        const resize = old && event.target.getAttribute("cursor") === "ew-resize";
        const next = old ? structuredClone(old) : {start: Math.max(start, Math.min(end - this.snap, start + Math.floor((first.x - KEYS) / PX * PPQ / this.snap) * this.snap)),
            duration: Math.min(this.snap, end - start), pitch: Math.max(0, Math.min(127, this.low + 35 - Math.floor((first.y - TOP) / ROW)))};
        const preview = shape("rect", {rx: 3, fill: "#fff", opacity: 0.65, "pointer-events": "none"}, svg);
        let heardPitch;
        const paint = () => {
            if (next.pitch !== heardPitch) { heardPitch = next.pitch; this.audition(next.pitch); }
            this.noteStatus.textContent = `${pitchName(next.pitch)} · bar ${Math.floor(next.start / this.data.roll.bar_ticks) + 1} · beat ${1 + (next.start % this.data.roll.bar_ticks) / PPQ} · ${next.duration / PPQ} beats long`;
            preview.setAttribute("x", KEYS + (Math.max(start, next.start) - start) / PPQ * PX);
            preview.setAttribute("y", TOP + (this.low + 35 - next.pitch) * ROW + 1);
            preview.setAttribute("width", (Math.min(end, next.start + next.duration) - Math.max(start, next.start)) / PPQ * PX);
            preview.setAttribute("height", ROW - 2);
        };
        paint(); svg.setPointerCapture(event.pointerId);
        svg.onpointermove = e => {
            const p = point(e);
            if (old) {
                const delta = Math.round((p.x - first.x) / PX * PPQ / this.snap) * this.snap;
                if (resize) next.duration = Math.max(this.snap, Math.min(this.data.roll.total_ticks - old.start, old.duration + delta));
                else {
                    next.start = Math.max(0, Math.min(this.data.roll.total_ticks - old.duration, old.start + delta));
                    next.pitch = Math.max(0, Math.min(127, old.pitch - Math.round((p.y - first.y) / ROW)));
                }
            } else next.duration = Math.min(end - next.start, Math.max(this.snap, Math.ceil((p.x - first.x) / PX * PPQ / this.snap) * this.snap));
            paint();
        };
        const finish = () => { svg.onpointermove = null; svg.onpointerup = null; svg.onpointercancel = null; };
        svg.onpointercancel = () => { finish(); this.draw(); };
        svg.onpointerup = () => {
            finish(); if (old && JSON.stringify(old) === JSON.stringify(next)) { this.draw(); return; }
            this.editor.commit(data => { if (index < 0) data.roll.tracks[this.track].push(next); else data.roll.tracks[this.track][index] = next; }).then(success => {
                if (success) { this.selectNotes([next]); this.draw(); }
            });
        };
    }
    selectNotes(notes) {
        this.selected = new Set(this.data.roll.tracks[this.track].flatMap((note, index) => notes.some(n => n.start === note.start && n.duration === note.duration && n.pitch === note.pitch) ? [index] : []));
    }
    selectionBox(notes, parent) {
        const {start, end} = this.bounds();
        notes = notes.filter(n => n.start < end && n.start + n.duration > start && n.pitch >= this.low && n.pitch < this.low + 36);
        if (!notes.length) return null;
        const x = KEYS + (Math.max(start, Math.min(...notes.map(n => n.start))) - start) / PPQ * this.zoom;
        const right = KEYS + (Math.min(end, Math.max(...notes.map(n => n.start + n.duration))) - start) / PPQ * this.zoom;
        const high = Math.max(...notes.map(n => n.pitch)), low = Math.min(...notes.map(n => n.pitch));
        return shape("rect", {x, y: TOP + (this.low + 35 - high) * ROW, width: right - x, height: (high - low + 1) * ROW,
            fill: "none", stroke: "#fff", "stroke-width": 1.5, "stroke-dasharray": "5 3", "pointer-events": "none"}, parent);
    }
    showSelection() {
        for (const {index, rect} of this.noteRects) {
            rect.setAttribute("stroke", this.selected.has(index) ? "white" : "#16727c");
            rect.setAttribute("stroke-width", this.selected.has(index) ? 2 : 1);
        }
        this.selectionOutline?.remove();
        this.selectionOutline = this.selectionBox([...this.selected].map(i => this.data.roll.tracks[this.track][i]), this.svg);
        this.noteStatus.textContent = this.selected.size ? `${this.selected.size} note${this.selected.size === 1 ? "" : "s"} selected` : "Shift-drag to select notes.";
    }
    boxSelect(event, point) {
        this.pause(); const svg = this.svg, first = point(event), before = new Set(this.selected), {start, end} = this.bounds();
        const marquee = shape("rect", {fill: "#43cfb1", "fill-opacity": 0.15, stroke: "#fff", "stroke-dasharray": "5 3", "pointer-events": "none"}, svg);
        let dragged = false;
        const update = e => {
            const p = point(e); dragged ||= Math.hypot(p.x - first.x, p.y - first.y) > 3;
            if (!dragged) return;
            const left = Math.max(KEYS, Math.min(first.x, p.x)), right = Math.min(Number(svg.getAttribute("width")), Math.max(first.x, p.x));
            const top = Math.max(TOP, Math.min(first.y, p.y)), bottom = Math.min(TOP + 36 * ROW, Math.max(first.y, p.y));
            for (const [name, value] of Object.entries({x: left, y: top, width: Math.max(0, right - left), height: Math.max(0, bottom - top)})) marquee.setAttribute(name, value);
            this.selected = new Set(before);
            this.data.roll.tracks[this.track].forEach((note, index) => {
                if (note.start >= end || note.start + note.duration <= start || note.pitch < this.low || note.pitch >= this.low + 36) return;
                const x = KEYS + (note.start - start) / PPQ * this.zoom, y = TOP + (this.low + 35 - note.pitch) * ROW;
                if (x < right && x + note.duration / PPQ * this.zoom > left && y < bottom && y + ROW > top) this.selected.add(index);
            });
            this.showSelection();
        };
        const finish = () => { svg.onpointermove = null; svg.onpointerup = null; svg.onpointercancel = null; if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId); marquee.remove(); };
        svg.setPointerCapture(event.pointerId); svg.onpointermove = update;
        svg.onpointerup = e => {
            update(e);
            if (!dragged) {
                const hit = event.target.getAttribute("data-note");
                if (hit !== null) { const index = Number(hit); this.selected.has(index) ? this.selected.delete(index) : this.selected.add(index); }
            }
            finish(); this.showSelection();
        };
        svg.onpointercancel = () => { finish(); this.selected = before; this.showSelection(); };
    }
    async commitSelection(notes) {
        const indices = [...this.selected], before = indices.map(i => ({...this.data.roll.tracks[this.track][i]}));
        const success = await this.editor.commit(data => { indices.forEach((index, i) => { data.roll.tracks[this.track][index] = notes[i]; }); });
        this.selectNotes(success ? notes : before); this.draw();
        return success;
    }
    moveSelection(event, point, anchor) {
        const svg = this.svg, first = point(event), indices = [...this.selected], before = indices.map(i => ({...this.data.roll.tracks[this.track][i]}));
        let next = before, preview, heardPitch;
        const minStart = Math.min(...before.map(n => n.start)), maxEnd = Math.max(...before.map(n => n.start + n.duration));
        const minPitch = Math.min(...before.map(n => n.pitch)), maxPitch = Math.max(...before.map(n => n.pitch));
        const update = e => {
            const p = point(e);
            const delta = Math.max(-minStart, Math.min(this.data.roll.total_ticks - maxEnd, Math.round((p.x - first.x) / this.zoom * PPQ / this.snap) * this.snap));
            const pitch = Math.max(-minPitch, Math.min(127 - maxPitch, -Math.round((p.y - first.y) / ROW)));
            next = before.map(n => ({...n, start: n.start + delta, pitch: n.pitch + pitch}));
            preview?.remove(); preview = shape("g", {"pointer-events": "none"}, svg);
            for (const note of next) this.selectionBox([note], preview);
            this.selectionBox(next, preview);
            const heard = next[indices.indexOf(anchor)].pitch;
            if (heard !== heardPitch) { heardPitch = heard; this.audition(heard); }
        };
        const finish = () => { svg.onpointermove = null; svg.onpointerup = null; svg.onpointercancel = null; if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId); preview?.remove(); };
        svg.setPointerCapture(event.pointerId); update(event); svg.onpointermove = update;
        svg.onpointerup = e => { update(e); finish(); if (JSON.stringify(before) !== JSON.stringify(next)) this.commitSelection(next); };
        svg.onpointercancel = () => { finish(); this.draw(); };
    }
    renderChords() {
        const {start, section} = this.bounds(), root = el("details", null, this.editor.piano);
        el("summary", "Harmony (optional)", root);
        root.open = this.harmonyOpen || false; root.ontoggle = () => { this.harmonyOpen = root.open; };
        const row = el("div", null, root); row.className = "chords";
        const roots = ["C", "C#", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
        for (let bar = 0; bar < section.bars; bar++) {
            const tick = start + bar * this.data.roll.bar_ticks, at = this.data.roll.chords.find(chord => chord.start === tick);
            const card = el("div", null, row);
            el("strong", `Bar ${section.start + bar + 1}`, card);
            el("div", at?.symbol || "No chord", card);
            const match = at?.symbol.match(/^([A-G][#b]?)(.*)$/);
            const set = symbol => this.editor.commit(data => {
                data.roll.chords = data.roll.chords.filter(c => c.start !== tick);
                if (symbol !== "—") data.roll.chords.push({start: tick, symbol});
            });
            const rootSelect = combo(card, "Root", [...new Set(["—", ...roots, ...(match ? [match[1]] : [])])], match?.[1] || "—", root => set(root === "—" ? root : root + quality.value));
            const qualities = [...new Set([...this.data.qualities, ...(match ? [match[2]] : [])])];
            const quality = combo(card, "Type", qualities.map(q => [q, q || "Major"]), match?.[2] || "", q => { if (rootSelect.value !== "—") set(rootSelect.value + q); });
        }
        if (this.data.roll.chords.some(c => c.start % this.data.roll.bar_ticks !== 0)) el("p", "Chord changes within a bar are preserved; edit their timing in Advanced ABC.", root).className = "help";
        button(root, "Remove chords (free harmony)", () => this.editor.commit(data => { data.roll.chords = []; }), true);
    }
    deleteNote() { if (this.selected.size) this.editor.commit(data => { data.roll.tracks[this.track] = data.roll.tracks[this.track].filter((_, i) => !this.selected.has(i)); }); }
    async nudge(key, octave) {
        if (!this.selected.size || this.editor.busy) return;
        const next = [...this.selected].map(index => {
            const note = {...this.data.roll.tracks[this.track][index]};
            if (key === "ArrowUp" || key === "ArrowDown") note.pitch += (key === "ArrowUp" ? 1 : -1) * (octave ? 12 : 1);
            else note.start += (key === "ArrowRight" ? 1 : -1) * this.snap;
            return note;
        });
        if (next.some(n => n.pitch < 0 || n.pitch > 127 || n.start < 0 || n.start + n.duration > this.data.roll.total_ticks)) return;
        if (await this.commitSelection(next)) this.audition(next[0].pitch);
    }
    fit() {
        const {start, end} = this.bounds();
        const notes = this.data.roll.tracks[this.track].filter(n => n.start < end && n.start + n.duration > start);
        if (notes.length) this.low = Math.max(0, Math.min(92, Math.floor((Math.min(...notes.map(n => n.pitch)) + Math.max(...notes.map(n => n.pitch)) - 35) / 2)));
        this.render(); this.scroll.scrollLeft = 0; this.scroll.scrollTop = 0; this.scroll.onscroll();
    }
    async audition(pitch) {
        if (pitch < 0 || pitch > 127) return;
        clearTimeout(this.auditionTimer);
        const audio = this.auditionAudio ||= new AudioContext(), version = this.auditionVersion = (this.auditionVersion || 0) + 1;
        await audio.resume(); if (this.auditionAudio !== audio || version !== this.auditionVersion) return;
        this.auditionVoice?.stop();
        this.auditionVoice = this.tone(audio, pitch, audio.currentTime, audio.currentTime + 0.25);
        this.auditionTimer = setTimeout(() => this.stopAudition(), 450);
    }
    stopAudition() {
        clearTimeout(this.auditionTimer); this.auditionAudio?.close(); this.auditionAudio = null; this.auditionVoice = null;
    }
    tone(audio, pitch, on, off, volume = 0.12, type = "triangle") {
        const oscillator = audio.createOscillator(), gain = audio.createGain();
        oscillator.type = type; oscillator.frequency.value = 440 * 2 ** ((pitch - 69) / 12);
        gain.gain.setValueAtTime(0, on); gain.gain.linearRampToValueAtTime(volume, on + Math.min(0.005, (off - on) / 2)); gain.gain.setTargetAtTime(0.001, off, 0.02);
        oscillator.connect(gain); gain.connect(audio.destination); oscillator.start(on); oscillator.stop(off + 0.15);
        return oscillator;
    }
    shift(data, at, delta) {
        for (const notes of Object.values(data.roll.tracks)) for (const note of notes) if (note.start >= at) note.start += delta;
        for (const chord of data.roll.chords) if (chord.start >= at) chord.start += delta;
    }
    checkBoundary(data, tick) {
        if (Object.values(data.roll.tracks).flat().some(n => n.start < tick && n.start + n.duration > tick)) throw new Error("A sustained note crosses this section boundary. Shorten it before changing sections.");
    }
    addBar() {
        const {end} = this.bounds(); this.editor.commit(data => { this.checkBoundary(data, end); this.shift(data, end, data.roll.bar_ticks); data.roll.sections[this.section].bars++; });
    }
    duplicateSection() {
        const {start, end, section} = this.bounds();
        this.editor.commit(data => {
            this.checkBoundary(data, start); this.checkBoundary(data, end);
            const copies = Object.fromEntries(Object.entries(data.roll.tracks).map(([name, notes]) => [name, notes.filter(n => n.start >= start && n.start < end).map(n => ({...n, start: n.start + end - start}))]));
            const chords = data.roll.chords.filter(c => c.start >= start && c.start < end).map(c => ({...c, start: c.start + end - start}));
            this.shift(data, end, end - start);
            for (const name of ["Vocal", "Ins"]) data.roll.tracks[name].push(...copies[name]);
            data.roll.chords.push(...chords); data.roll.sections.splice(this.section + 1, 0, {...section});
        });
    }
    removeSection() {
        const {start, end} = this.bounds();
        this.editor.commit(data => {
            this.checkBoundary(data, start); this.checkBoundary(data, end);
            for (const name of ["Vocal", "Ins"]) data.roll.tracks[name] = data.roll.tracks[name].filter(n => n.start < start || n.start >= end);
            data.roll.chords = data.roll.chords.filter(c => c.start < start || c.start >= end);
            this.shift(data, end, start - end); data.roll.sections.splice(this.section, 1);
        });
    }
    moveSection(delta) {
        const current = this.section, other = current + delta;
        if (other < 0 || other >= this.data.roll.sections.length) return;
        this.editor.commit(data => {
            const sections = data.roll.sections, ranges = sections.map(s => ({start: s.start * data.roll.bar_ticks, end: (s.start + s.bars) * data.roll.bar_ticks}));
            const order = sections.map((_, i) => i); [order[current], order[other]] = [order[other], order[current]];
            let cursor = 0; const shifts = {};
            for (const i of order) { shifts[i] = cursor - ranges[i].start; cursor += ranges[i].end - ranges[i].start; }
            for (const notes of Object.values(data.roll.tracks)) for (const note of notes) {
                const i = ranges.findIndex(r => note.start >= r.start && note.start < r.end);
                if (note.start + note.duration > ranges[i].end) throw new Error("Shorten notes crossing section boundaries before reordering.");
                note.start += shifts[i];
            }
            for (const chord of data.roll.chords) { const i = ranges.findIndex(r => chord.start >= r.start && chord.start < r.end); chord.start += shifts[i]; }
            data.roll.sections = order.map(i => sections[i]);
        });
    }
    updateCursor() {
        if (!this.cursor || !this.data?.grid_available) return;
        const {start, end} = this.bounds();
        const x = KEYS + (Math.max(start, Math.min(end, this.position)) - start) / PPQ * this.zoom;
        this.cursor.setAttribute("x1", x); this.cursor.setAttribute("x2", x); this.cursorHandle.setAttribute("x", x - 5);
    }
    scrub(event, point) {
        const svg = this.svg, {start, end} = this.bounds(), playing = this.playing;
        this.pause(); const before = this.position;
        const seek = e => { this.position = Math.max(start, Math.min(end, start + (point(e).x - KEYS) / this.zoom * PPQ)); this.updateCursor(); };
        const finish = () => {
            svg.onpointermove = null; svg.onpointerup = null; svg.onpointercancel = null;
            if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
            if (playing) this.play();
        };
        svg.setPointerCapture(event.pointerId); seek(event);
        svg.onpointermove = seek; svg.onpointerup = e => { seek(e); finish(); };
        svg.onpointercancel = () => { this.position = before; this.updateCursor(); finish(); };
    }
    togglePlayback() { if (!this.data?.grid_available) return; return this.playing ? this.pause() : this.play(); }
    async play() {
        this.pause(); this.stopAudition(); const {start, end} = this.bounds();
        if (this.position < start || this.position >= end) this.position = start;
        const audio = new AudioContext(); this.audio = audio; this.playing = true;
        this.playButton.textContent = "Pause";
        await audio.resume(); if (this.audio !== audio) return;
        const now = audio.currentTime + 0.04, seconds = 60 / this.data.bpm / PPQ, from = this.position;
        this.startedAt = now; this.startedFrom = from; this.secondsPerTick = seconds;
        for (const note of this.data.roll.tracks[this.track]) {
            const a = Math.max(from, note.start), b = Math.min(end, note.start + note.duration); if (b <= a) continue;
            const on = now + (a - from) * seconds, off = now + (b - from) * seconds;
            this.tone(audio, note.pitch, on, off);
        }
        if (this.metronome) {
            const beat = PPQ * 4 / Number(this.data.meter.split("/")[1]);
            for (let tick = Math.ceil(from / beat) * beat; tick < end; tick += beat) {
                const on = now + (tick - from) * seconds;
                this.tone(audio, tick % this.data.roll.bar_ticks === 0 ? 96 : 84, on, on + 0.015, 0.04, "square");
            }
        }
        const animate = () => {
            if (this.audio !== audio) return;
            this.position = Math.min(end, from + Math.max(0, audio.currentTime - now) / seconds);
            this.updateCursor();
            if (this.position >= end) { this.pause(); return; }
            this.animation = requestAnimationFrame(animate);
        };
        this.animation = requestAnimationFrame(animate);
    }
    pause() {
        if (this.audio) {
            if (this.startedAt !== undefined) this.position = Math.min(this.bounds().end, this.startedFrom + Math.max(0, this.audio.currentTime - this.startedAt) / this.secondsPerTick);
            this.audio.close(); this.audio = null;
        }
        this.startedAt = undefined; this.playing = false; cancelAnimationFrame(this.animation);
        if (this.playButton) this.playButton.textContent = "Play";
        this.updateCursor();
    }
    stop() { this.pause(); this.stopAudition(); this.position = this.data?.grid_available ? this.bounds().start : 0; this.updateCursor(); }
}
