import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { PianoRoll, el, button } from "./piano_roll.js";

const style = document.createElement("style");
style.textContent = `
.fl-yue2-editor{height:100%;padding:12px;box-sizing:border-box;background:#29003d;color:#f5edf8;font:13px system-ui;border:1px solid #16727c;border-radius:8px}
.fl-yue2-content{display:flow-root}
.fl-yue2-editor *{box-sizing:border-box}.fl-yue2-editor .row{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.fl-yue2-editor label{display:flex;align-items:center;gap:5px}.fl-yue2-editor input,.fl-yue2-editor select,.fl-yue2-editor textarea{background:#180024;color:white;border:1px solid #765287;border-radius:5px;padding:7px;min-width:0}
.fl-yue2-editor input[type=number]{width:70px}.fl-yue2-editor button{background:#16727c;color:white;border:0;border-radius:5px;padding:8px 11px;cursor:pointer}.fl-yue2-editor button.secondary{background:#553064}.fl-yue2-editor button:disabled{opacity:.4;cursor:default}
.fl-yue2-editor .status{padding:8px;border-left:3px solid #52cfb1;margin:8px 0}.fl-yue2-editor .status.error{border-color:#ff9292;color:#ffbcbc}.fl-yue2-editor .help{font-size:12px;color:#d5bfdf;line-height:1.5;margin:8px 0}
.fl-yue2-editor .roll{overflow-x:auto;border:1px solid #715084;border-radius:6px;overscroll-behavior:contain}.fl-yue2-editor svg{display:block;touch-action:none;user-select:none}.fl-yue2-editor textarea{width:100%;min-height:430px;font:13px monospace}.fl-yue2-editor [hidden]{display:none!important}
.fl-yue2-editor .chords{display:flex;flex-wrap:wrap;gap:6px;padding:6px 0}.fl-yue2-editor .chords label{flex-direction:column;font-size:11px}.fl-yue2-editor .chords select{max-width:130px}
.fl-yue2-expanded{position:fixed;inset:3vh 3vw;width:94vw;height:94vh;overflow:auto;z-index:10000;box-shadow:0 0 0 4vh #0009}
`;
document.head.append(style);

class ScoreEditor {
    constructor(node, source) {
        this.node = node; this.source = source; this.revision = 0;
        this.undo = []; this.redo = [];
        this.root = el("div"); this.root.className = "fl-yue2-editor"; this.root.tabIndex = 0;
        this.content = el("div", null, this.root); this.content.className = "fl-yue2-content";
        this.widgetHeight = 900;
        this.root.addEventListener("pointerdown", event => {
            if (this.expanded || event.button !== 1) return;
            event.preventDefault();
            event.stopPropagation();
            // ComfyUI captures this active pointer and handles the rest of the drag.
            app.canvas.canvas.dispatchEvent(new PointerEvent("pointerdown", {
                bubbles: true, cancelable: true,
                pointerId: event.pointerId, pointerType: event.pointerType, isPrimary: event.isPrimary,
                button: event.button, buttons: event.buttons,
                clientX: event.clientX, clientY: event.clientY,
                ctrlKey: event.ctrlKey, metaKey: event.metaKey, shiftKey: event.shiftKey, altKey: event.altKey,
            }));
        }, {capture: true});
        this.root.addEventListener("wheel", event => {
            if (this.expanded) return;
            event.preventDefault();
            event.stopPropagation();
            app.canvas.canvas.dispatchEvent(new WheelEvent("wheel", {
                bubbles: true, cancelable: true,
                clientX: event.clientX, clientY: event.clientY,
                deltaX: event.deltaX, deltaY: event.deltaY, deltaZ: event.deltaZ, deltaMode: event.deltaMode,
                ctrlKey: event.ctrlKey, metaKey: event.metaKey, shiftKey: event.shiftKey, altKey: event.altKey,
            }));
        }, {capture: true, passive: false});
        this.root.addEventListener("keydown", event => {
            event.stopPropagation();
            if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
            if (event.key === "Delete" || event.key === "Backspace") { event.preventDefault(); this.roll.deleteNote(); }
            if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") { event.preventDefault(); this.restore(event.shiftKey); }
            if (event.code === "Space") { event.preventDefault(); if (!event.repeat) this.roll.togglePlayback(); }
            if (event.key === "Escape") this.expand(false);
            if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) {
                event.preventDefault(); this.roll.nudge(event.key, event.shiftKey);
            }
        });
        const toolbar = el("div", null, this.content); toolbar.className = "row";
        this.pianoButton = button(toolbar, "Piano roll", () => this.showPiano());
        button(toolbar, "New blank score", () => this.commit(data => {
            data.roll.tracks = {Vocal: [], Ins: []}; data.roll.chords = []; data.roll.sections = [{name: "verse", bars: 4}];
        }), true);
        button(toolbar, "Advanced ABC", () => { this.roll.pause(); this.piano.hidden = true; this.raw.hidden = false; }, true);
        button(toolbar, "Import ABC", () => this.file.click(), true);
        button(toolbar, "Export ABC", () => this.export(), true);
        this.undoButton = button(toolbar, "Undo", () => this.restore(false), true);
        this.redoButton = button(toolbar, "Redo", () => this.restore(true), true);
        button(toolbar, "Expand / close", () => this.expand(!this.expanded), true);
        this.connection = el("div", null, this.content); this.connection.className = "row";
        this.connectionText = el("span", null, this.connection);
        this.file = el("input", null, this.content); this.file.type = "file"; this.file.accept = ".abc,.txt"; this.file.hidden = true;
        this.file.onchange = async () => {
            const file = this.file.files[0]; if (!file) return;
            if (file.size > 200000) return this.status("Maximum score file size is 200 KB.", true);
            this.undo.push(this.source.value); this.redo = []; this.write(await file.text());
            if (await this.validate()) this.showPiano(); this.file.value = "";
        };
        this.message = el("div", "Loading score…", this.content); this.message.className = "status";
        this.piano = el("div", null, this.content); this.roll = new PianoRoll(this);
        this.raw = el("textarea", null, this.content); this.raw.hidden = true; this.raw.spellcheck = false;
        this.raw.setAttribute("aria-label", "Advanced ABC source");
        this.raw.oninput = () => {
            this.revision++; this.roll.pause(); this.pianoButton.disabled = true;
            this.write(this.raw.value); clearTimeout(this.timer); this.timer = setTimeout(() => this.validate(), 450);
        };
        el("p", "Preview plays a simple melody sound. YuE2 creates the finished instruments and production when you queue the workflow.", this.content).className = "help";
        this.observer = new ResizeObserver(() => this.scheduleLayout());
        this.observer.observe(this.content);
        this.load();
    }
    scheduleLayout() {
        cancelAnimationFrame(this.layoutFrame);
        this.layoutFrame = requestAnimationFrame(() => {
            const width = this.content.clientWidth;
            if (!width) return;
            if (width !== this.contentWidth) {
                this.contentWidth = width;
                if (this.roll.scroll && this.data?.grid_available) this.roll.draw();
            }
            if (this.expanded) return;
            this.widgetHeight = Math.max(900, this.content.offsetHeight + 26);
            const height = this.widgetHeight + 70;
            if (this.node.size[1] < height) this.node.setSize([this.node.size[0], height]);
            this.node.graph?.setDirtyCanvas(true, true);
        });
    }
    status(text, error = false) { this.message.textContent = text; this.message.className = "status" + (error ? " error" : ""); }
    write(text) { this.source.value = text; this.raw.value = text; this.node.graph?.setDirtyCanvas(true, true); }
    get connected() { return this.node.inputs?.some(input => input.name === "incoming_score_abc" && input.link != null); }
    get sourceHash() { return this.node.widgets.find(widget => widget.name === "source_score_hash"); }
    connectionChanged() {
        const link = this.node.inputs?.find(input => input.name === "incoming_score_abc")?.link ?? null;
        if (link !== this.link) { this.link = link; this.revision++; this.roll.pause(); }
        this.connectionText.textContent = this.connected ? "Editable score. Queue to load upstream changes; unchanged input keeps your edits." : "Edits are saved in this workflow.";
    }
    async receive(text, sourceHash) {
        if (!this.connected) return;
        if (sourceHash && sourceHash === this.sourceHash.value) return;
        const revision = ++this.revision; this.roll.stop();
        try {
            const result = await this.request("/fl_yue2/score/validate", {score_abc: text});
            if (revision !== this.revision || !this.connected) return;
            this.sourceHash.value = sourceHash || ""; this.undo = []; this.redo = [];
            this.accept(result); this.showPiano();
        } catch (error) { if (revision === this.revision) this.status(error.message, true); }
    }
    expand(expanded) {
        if (expanded === !!this.expanded) return;
        this.expanded = expanded;
        if (expanded) { this.home = this.root.parentNode; document.body.append(this.root); this.root.classList.add("fl-yue2-expanded"); }
        else { this.root.classList.remove("fl-yue2-expanded"); this.home?.append(this.root); }
        this.root.focus();
        this.scheduleLayout();
    }
    load() { this.roll.stop(); this.undo = []; this.redo = []; this.raw.value = this.source.value; this.connectionChanged(); this.validate(); }
    async request(path, body) {
        const response = await api.fetchApi(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
        const result = await response.json(); if (!response.ok) throw new Error(result.error || "Score validation failed."); return result;
    }
    async validate() {
        clearTimeout(this.timer); this.roll.pause(); const revision = ++this.revision;
        try {
            const result = await this.request("/fl_yue2/score/validate", {score_abc: this.source.value});
            if (revision !== this.revision) return false; this.accept(result); return true;
        } catch (error) {
            if (revision === this.revision) { this.status(error.message, true); this.pianoButton.disabled = true; this.piano.hidden = true; this.raw.hidden = false; }
            return false;
        }
    }
    accept(result) {
        this.data = result;
        this.write(result.abc);
        this.pianoButton.disabled = !result.grid_available;
        this.status(`${result.bars} bars · ${result.bpm} BPM · about ${Math.round(result.seconds)} seconds`);
        this.undoButton.disabled = !this.undo.length; this.redoButton.disabled = !this.redo.length;
        if (result.grid_available) this.roll.render();
        else { this.piano.hidden = true; this.raw.hidden = false; this.status("Key or meter changes within the song: original notation preserved in Advanced ABC."); }
    }
    showPiano() { if (!this.data || this.pianoButton.disabled) return; this.piano.hidden = false; this.raw.hidden = true; }
    async commit(change) {
        if (this.busy) return;
        this.roll.pause(); this.busy = true;
        const before = this.source.value, draft = structuredClone(this.data), revision = ++this.revision;
        try {
            change(draft); const result = await this.request("/fl_yue2/score/build", draft);
            if (revision !== this.revision) return;
            this.undo.push(before); if (this.undo.length > 50) this.undo.shift(); this.redo = []; this.accept(result); return true;
        } catch (error) {
            if (revision === this.revision) { this.roll.render(); this.status(error.message, true); }
            return false;
        } finally { this.busy = false; }
    }
    async restore(redo) {
        const from = redo ? this.redo : this.undo, to = redo ? this.undo : this.redo;
        if (!from.length || this.busy) return; this.roll.stop(); to.push(this.source.value); this.write(from.pop()); await this.validate();
    }
    async export() {
        if (!await this.validate()) return;
        const url = URL.createObjectURL(new Blob([this.data.abc], {type: "text/plain"}));
        const link = el("a"); link.href = url; link.download = "yue2-score.abc"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
    destroy() { this.expand(false); this.observer.disconnect(); cancelAnimationFrame(this.layoutFrame); this.revision++; clearTimeout(this.timer); this.roll.stop(); }
}

app.registerExtension({
    name: "FL.YuE2.ScoreEditor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FL_YuE2_ScoreEditor") return;
        const resized = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            resized?.apply(this, arguments);
            size[0] = this.size[0] = Math.max(840, size[0]);
            size[1] = this.size[1] = Math.max((this.yue2Editor?.widgetHeight ?? 900) + 70, size[1]);
            this.yue2Editor?.scheduleLayout();
        };
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            created?.apply(this, arguments);
            const source = this.widgets.find(w => w.name === "score_abc");
            const sourceHash = this.widgets.find(w => w.name === "source_score_hash")
                || this.addWidget("text", "source_score_hash", "", () => {});
            sourceHash.hidden = true; sourceHash.computeSize = () => [0, -4]; sourceHash.type = "converted-widget";
            if (sourceHash.element) sourceHash.element.style.display = "none";
            if (sourceHash.inputEl) sourceHash.inputEl.hidden = true;
            source.hidden = true; source.computeSize = () => [0, -4]; source.type = "converted-widget";
            if (source.element) source.element.style.display = "none"; if (source.inputEl) source.inputEl.hidden = true;
            this.yue2Editor = new ScoreEditor(this, source);
            const widget = this.addDOMWidget("yue2_score_editor", "custom", this.yue2Editor.root, {
                serialize: false,
                getMinHeight: () => this.yue2Editor.widgetHeight,
                getHeight: () => this.yue2Editor.widgetHeight,
            });
            widget.computeSize = () => [800, this.yue2Editor.widgetHeight]; this.setSize([840, 970]);
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            configured?.apply(this, arguments);
            if (!this.inputs?.some(input => input.name === "incoming_score_abc")) this.addInput("incoming_score_abc", "STRING");
            this.setSize([Math.max(840, this.size[0]), Math.max(970, this.size[1])]);
            this.yue2Editor?.load();
        };
        const removed = nodeType.prototype.onRemoved;
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            executed?.apply(this, arguments);
            if (message.score_abc?.[0]) this.yue2Editor?.receive(message.score_abc[0], message.source_score_hash?.[0]);
        };
        const connections = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            connections?.apply(this, arguments);
            this.yue2Editor?.connectionChanged();
            if (this.yue2Editor && !this.yue2Editor.connected) { this.yue2Editor.validate().then(() => this.yue2Editor?.showPiano()); }
        };
        nodeType.prototype.onRemoved = function () { this.yue2Editor?.destroy(); return removed?.apply(this, arguments); };
    },
});
