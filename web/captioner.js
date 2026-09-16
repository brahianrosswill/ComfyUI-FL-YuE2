import {app} from "../../scripts/app.js";
import {api} from "../../scripts/api.js";

const css = document.createElement("style");
css.textContent = `
.yue2-captioner{height:100%;box-sizing:border-box;display:flex;flex-direction:column;gap:10px;padding:12px;background:#111820;border:1px solid #304251;border-radius:10px;color:#e3edf5;font:12px system-ui;overflow:auto}
.yue2-captioner *{box-sizing:border-box}.yue2-captioner section{padding:12px;background:#19232e;border:1px solid #304251;border-radius:8px;min-height:0}
.yue2-captioner h3{font-size:13px;color:#70ddde;margin:0 0 8px}.yue2-captioner p{margin:6px 0;color:#abbaca;line-height:1.4}
.yue2-captioner textarea,.yue2-captioner select{width:100%;background:#101820;border:1px solid #405266;border-radius:5px;color:#edf5fa;padding:9px;font:12px system-ui}
.yue2-captioner textarea{resize:vertical;min-height:80px}.yue2-captioner .prompt{height:140px}.yue2-captioner .actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:8px 0}
.yue2-captioner button{background:#166673;border:1px solid #248594;color:white;border-radius:5px;padding:7px 12px;font:inherit;cursor:pointer}.yue2-captioner button:disabled{opacity:.45;cursor:default}
.yue2-captioner progress{width:100%;height:12px;accent-color:#45d2ca}.yue2-captioner .status{overflow-wrap:anywhere;min-height:17px}.yue2-captioner .error{color:#ffad9a}
.yue2-captioner .results{flex:1;min-height:240px;overflow:auto}.yue2-captioner audio{width:100%;height:36px;margin:8px 0}.yue2-captioner label{display:block;margin:8px 0 4px;color:#b9d2e3}
.yue2-captioner .fields{display:grid;grid-template-columns:1fr 1fr;gap:12px}.yue2-captioner .fields textarea{height:145px}.yue2-captioner .warning{color:#ffca91;white-space:pre-wrap}.yue2-captioner .filename{overflow-wrap:anywhere}
`;
document.head.append(css);

function el(tag, parent, text) {
    const result = document.createElement(tag);
    if (text != null) result.textContent = text;
    parent?.append(result);
    return result;
}

async function json(path, body) {
    const response = await api.fetchApi(path, body ? {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)} : {});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Request failed");
    return result;
}

export class CaptionPanel {
    constructor(node) {
        this.node = node;
        this.busy = false;
        this.promptWidget = node.widgets.find(w => w.name === "instructions");
        for (const widget of node.widgets.filter(w => ["instructions", "test_random"].includes(w.name))) {
            widget.type = "converted-widget";
            widget.computeSize = () => [0, -4];
            widget.hidden = true;
            if (widget.element) widget.element.style.display = "none";
        }
        this.root = el("div"); this.root.className = "yue2-captioner";
        for (const event of ["pointerdown", "keydown", "wheel"]) this.root.addEventListener(event, e => e.stopPropagation());
        const top = el("section", this.root);
        el("h3", top, "Music captioning / System prompt");
        el("p", top, "Add your directions to the built-in music and lyric instructions. Both folder runs and random tests use this prompt.");
        this.prompt = el("textarea", top); this.prompt.className = "prompt"; this.prompt.setAttribute("aria-label", "Captioner system prompt");
        this.prompt.placeholder = "Describe the musical details you want Gemini to focus on…";
        this.prompt.oninput = () => { this.promptWidget.value = this.prompt.value; this.node.graph?.change(); };
        const actions = el("div", top); actions.className = "actions";
        this.folder = el("button", actions, "Process folder"); this.folder.onclick = () => this.queue(false);
        this.test = el("button", actions, "Test random track"); this.test.onclick = () => this.queue(true);
        this.refresh = el("button", actions, "Load saved captions"); this.refresh.onclick = () => this.load(false).catch(e => this.status(e.message, true));
        this.count = el("p", top, "Folder progress · ready");
        this.progress = el("progress", top); this.progress.max = 1; this.progress.value = 0; this.progress.setAttribute("aria-label", "Captioning progress");
        this.message = el("p", top, "Uses your Google API key above. Random tests leave dataset captions unchanged."); this.message.className = "status"; this.message.setAttribute("role", "status");
        this.bottom = el("section", this.root); this.bottom.className = "results";
        el("h3", this.bottom, "Listen & inspect");
        this.resultLabel = el("p", this.bottom, "Test a random track to see its audio, style caption and lyrics here.");
        this.select = el("select", this.bottom); this.select.hidden = true; this.select.setAttribute("aria-label", "Captioned song"); this.select.onchange = () => this.showSong(Number(this.select.value));
        this.content = el("div", this.bottom);
        this.listener = event => {
            const value = event.detail;
            if (String(value.node) !== String(node.id) || value.operation && value.operation !== "caption") return;
            if (value.job !== this.job && value.job) { this.job = value.job; this.setBusy(true); }
            if (value.type === "status") this.status(value.message);
            if (value.type === "progress") {
                if (value.test_random != null) this.testing = value.test_random;
                this.progress.max = value.max_steps; this.progress.value = value.step;
                this.count.textContent = `${this.testing ? "Random test" : "Folder progress"} · ${value.step} / ${value.max_steps} tracks · ${Math.round(value.step / value.max_steps * 100)}%`;
            }
            if (value.type === "error") { this.setBusy(false); this.status(value.message, true); }
            if (value.type === "complete") { this.setBusy(false); this.status("Complete"); }
        };
        this.failure = event => {
            if (event.detail.prompt_id !== this.promptId && String(event.detail.node_id) !== String(node.id)) return;
            this.setBusy(false); this.status(event.type === "execution_interrupted" ? "Captioning cancelled." : event.detail.exception_message || "Captioning failed.", true);
        };
        this.queueStatus = async () => {
            if (!this.busy || !this.promptId) return;
            const id = this.promptId;
            try {
                const queue = await json("/queue");
                if (id === this.promptId && this.busy && ![...queue.queue_running, ...queue.queue_pending].some(item => item[1] === id)) {
                    this.setBusy(false); this.status("Captioner left the queue.");
                }
            } catch { /* A reconnect will deliver the next queue status. */ }
        };
        api.addEventListener("fl_yue2.training", this.listener);
        api.addEventListener("execution_error", this.failure);
        api.addEventListener("execution_interrupted", this.failure);
        api.addEventListener("status", this.queueStatus);
        this.syncPrompt();
    }
    syncPrompt() {
        this.prompt.value = this.promptWidget.value || "";
        this.prompt.disabled = this.node.inputs?.some(input => input.name === "instructions" && input.link != null) || false;
    }
    status(message, error = false) { this.message.textContent = message; this.message.classList.toggle("error", error); }
    setBusy(busy) { this.busy = busy; this.folder.disabled = this.test.disabled = busy; }
    async queue(test) {
        if (this.busy) return;
        this.testing = test;
        this.promptId = null;
        this.setBusy(true); this.status("Queuing captioner…");
        this.progress.value = 0; this.count.textContent = test ? "Random test · queued" : "Folder progress · queued";
        try {
            if (!this.node.widgets.some(widget => widget.name === "test_random")) throw new Error("Restart ComfyUI after the current run finishes to enable the new captioner actions.");
            if (this.node.inputs?.some(input => input.link != null)) throw new Error("Use ComfyUI Queue for connected inputs. Panel actions use the values on this node.");
            const inputs = {};
            for (const widget of this.node.widgets) {
                if (widget.options?.serialize === false || widget.name === "test_random") continue;
                inputs[widget.name] = widget.value;
            }
            inputs.test_random = test;
            const result = await api.queuePrompt(0, {output:{[this.node.id]:{class_type:"FL_YuE2_GeminiMusicCaptioner", inputs}}, workflow:app.graph.serialize()});
            this.promptId = result.prompt_id;
            this.status(test ? "Random test queued. Dataset captions will not be changed." : "Folder queued. Existing captions follow replace_existing above.");
        } catch (error) { this.setBusy(false); this.status(error.message || "Could not queue captioner.", true); }
    }
    async load(test) {
        const id = this.node.properties[test ? "yue2_caption_test" : "yue2_captions"];
        if (!id) { this.status("No saved captions yet. Process the folder or test a random track."); return; }
        const request = this.loadRequest = (this.loadRequest || 0) + 1;
        const data = await json(`/fl_yue2/captions/${id}`);
        if (request !== this.loadRequest) return;
        this.resultId = id; this.songs = data.songs; this.resultIsTest = test;
        this.resultLabel.textContent = test ? "Random test · preview only · dataset unchanged" : `${data.songs.length} saved tracks · captions are automatically accepted and editable`;
        this.select.replaceChildren();
        data.songs.forEach((song, index) => { const option = el("option", this.select, song.name); option.value = index; });
        this.select.hidden = data.songs.length < 2;
        this.showSong(0);
    }
    clearAudio() { this.content.querySelectorAll("audio").forEach(audio => { audio.pause(); audio.removeAttribute("src"); audio.load(); }); }
    showSong(index) {
        this.clearAudio(); this.content.replaceChildren();
        const song = this.songs[index]; if (!song) return;
        el("p", this.content, song.name).className = "filename";
        const audio = el("audio", this.content); audio.controls = true; audio.preload = "metadata";
        audio.src = api.apiURL(`/fl_yue2/captions/${this.resultId}/audio/${encodeURIComponent(song.name)}`);
        audio.setAttribute("aria-label", song.name);
        const fields = el("div", this.content); fields.className = "fields";
        const values = {};
        for (const [key, title] of [["style", "Style caption"], ["lyrics", "Lyrics"]]) {
            const column = el("div", fields); const label = el("label", column, title);
            const text = el("textarea", label); text.value = song[key]; text.setAttribute("aria-label", title); text.readOnly = this.resultIsTest;
            text.placeholder = key === "lyrics" ? "No lyrics / instrumental" : "No style requested"; values[key] = text;
        }
        if (song.uncertainty) el("p", this.content, song.uncertainty).className = "warning";
        if (!this.resultIsTest) {
            const id = this.resultId;
            const save = el("button", this.content, "Save caption edits");
            save.onclick = async () => {
                save.disabled = true;
                try {
                    await json("/fl_yue2/captions/review", {identifier:id, name:song.name, style:values.style.value, lyrics:values.lyrics.value});
                    song.style = values.style.value; song.lyrics = values.lyrics.value; this.status("Caption edits saved.");
                } catch (error) { this.status(error.message, true); }
                finally { save.disabled = false; }
            };
        }
    }
    destroy() {
        this.loadRequest = (this.loadRequest || 0) + 1;
        api.removeEventListener("fl_yue2.training", this.listener);
        api.removeEventListener("execution_error", this.failure);
        api.removeEventListener("execution_interrupted", this.failure);
        api.removeEventListener("status", this.queueStatus);
        this.clearAudio(); this.root.remove();
    }
}

app.registerExtension({
    name:"FL.YuE2.Captioner",
    beforeRegisterNodeDef(type, data) {
        if (data.name !== "FL_YuE2_GeminiMusicCaptioner") return;
        const created = type.prototype.onNodeCreated, configured = type.prototype.onConfigure, executed = type.prototype.onExecuted, removed = type.prototype.onRemoved, connections = type.prototype.onConnectionsChange;
        type.prototype.onNodeCreated = function() {
            created?.apply(this, arguments); this.properties ||= {};
            this.yue2Captioner = new CaptionPanel(this);
            this.addDOMWidget("yue2_captioner", "custom", this.yue2Captioner.root, {serialize:false, hideOnZoom:false, getMinHeight:() => 570});
            this.setSize([Math.max(650, this.size[0]), Math.max(790, this.size[1])]);
        };
        type.prototype.onConfigure = function() {
            configured?.apply(this, arguments);
            this.yue2Captioner?.syncPrompt();
            this.setSize([Math.max(650, this.size[0]), Math.max(790, this.size[1])]);
            if (this.properties.yue2_caption_test || this.properties.yue2_captions) this.yue2Captioner?.load(!!this.properties.yue2_caption_test).catch(e => this.yue2Captioner.status(e.message, true));
        };
        type.prototype.onConnectionsChange = function() { connections?.apply(this, arguments); this.yue2Captioner?.syncPrompt(); };
        type.prototype.onExecuted = function(message) {
            executed?.apply(this, arguments);
            const test = !!message.caption_test;
            const id = (message.caption_test || message.captions)?.[0];
            if (id) {
                this.properties[test ? "yue2_caption_test" : "yue2_captions"] = id;
                this.yue2Captioner.setBusy(false);
                this.yue2Captioner.load(test).catch(e => this.yue2Captioner.status(e.message, true));
            }
        };
        type.prototype.onRemoved = function() { this.yue2Captioner?.destroy(); removed?.apply(this, arguments); };
    }
});
