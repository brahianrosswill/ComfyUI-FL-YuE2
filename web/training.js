import {app} from "../../scripts/app.js";
import {api} from "../../scripts/api.js";

const css = document.createElement("style");
css.textContent = `.yue2-training{background:#200c2e;color:#f1eaf5;padding:14px;font:13px system-ui;border:1px solid #16727c;border-radius:9px;box-sizing:border-box;height:100%;overflow:auto}
.yue2-training h3{margin:0 0 8px;color:#65ded1;font-size:15px}.yue2-training progress{width:100%;accent-color:#46c6b6}.yue2-training canvas{width:100%;height:125px;background:#140a1c;border-radius:6px;margin:10px 0}
.yue2-training textarea{box-sizing:border-box;width:100%;background:#140a1c;color:#fff;border:1px solid #67516f;border-radius:5px;padding:8px;margin:6px 0;min-height:72px}
.yue2-training button{background:#16727c;border:0;border-radius:5px;padding:8px 12px;color:white;cursor:pointer;margin:5px 5px 5px 0}.yue2-training article{padding:10px;border:1px solid #57425f;border-radius:7px;margin:8px 0}.yue2-training audio{width:100%;height:34px}.yue2-training .warning{color:#ffcc9c}.yue2-training small{display:block;color:#bfaec8;margin:6px 0}`;
css.textContent += `
 .yue2-trainer-widget { --primary: #06b6d4; --primary-glow: rgba(6, 182, 212, 0.4); --secondary: #8b5cf6; --success: #22c55e; --danger: #ef4444; --warning: #f59e0b; --bg-dark: #0f0f12; --bg-card: #18181b; --bg-elevated: #1f1f23; --border: #27272a; --border-hover: #3f3f46; --text-primary: #fafafa; --text-secondary: #a1a1aa; --text-muted: #71717a; background: var(--bg-card); border-radius: 12px; border: 1px solid var(--border); overflow: hidden; position: relative; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; color: var(--text-primary); box-sizing: border-box; height: 100%; min-height: 300px; display: flex; flex-direction: column; }
 .yue2-trainer-widget * { box-sizing: border-box; }
 .yue2-trainer-header { display: flex; justify-content: space-between; align-items: center; padding: 6px 10px; background: var(--bg-elevated); border-bottom: 1px solid var(--border); flex-shrink: 0; }
 .yue2-trainer-title { font-size: 11px; font-weight: 600; color: var(--text-primary); display: flex; align-items: center; gap: 8px; }
 .yue2-trainer-badge { padding: 2px 8px; background: var(--primary); color: white; border-radius: 10px; font-size: 10px; font-weight: 500; }
 .yue2-trainer-badge.idle { background: var(--text-muted); }
 .yue2-trainer-badge.training { background: var(--success); animation: yue2-trainer-pulse 2s infinite; }
 @keyframes yue2-trainer-pulse { 0%, 100% { opacity: 1; }
 50% { opacity: 0.7; }
 }
 .yue2-trainer-content { flex: 1; display: flex; flex-direction: column; padding: 10px; gap: 8px; overflow: hidden; }
 .yue2-trainer-stats { display: flex; gap: 12px; justify-content: center; align-items: baseline; flex-shrink: 0; }
 .yue2-trainer-stat { display: flex; align-items: baseline; gap: 4px; }
 .yue2-trainer-stat-label { font-size: 9px; color: var(--text-muted); text-transform: uppercase; }
 .yue2-trainer-stat-value { font-size: 11px; font-weight: 600; color: var(--primary); font-variant-numeric: tabular-nums; }
 .yue2-trainer-progress-section { background: var(--bg-elevated); border-radius: 6px; padding: 8px 10px; flex-shrink: 0; }
 .yue2-trainer-progress-header { display: flex; justify-content: space-between; margin-bottom: 4px; }
 .yue2-trainer-progress-label { font-size: 9px; color: var(--text-secondary); }
 .yue2-trainer-progress-value { font-size: 9px; color: var(--text-primary); font-weight: 500; }
 .yue2-trainer-progress-bar { height: 4px; background: var(--bg-dark); border-radius: 2px; overflow: hidden; }
 .yue2-trainer-progress-fill { height: 100%; background: linear-gradient(90deg, var(--primary), var(--secondary)); border-radius: 2px; transition: width 0.3s ease; width: 0%; }
 .yue2-trainer-chart-section { background: var(--bg-elevated); border-radius: 6px; padding: 8px 10px; flex: 1; min-height: 80px; display: flex; flex-direction: column; overflow: hidden; }
 .yue2-trainer-chart-header { font-size: 9px; color: var(--text-secondary); margin-bottom: 4px; }
 .yue2-trainer-chart-canvas { width: 100%; height: 100%; display: block; }
 .yue2-trainer-status { background: var(--bg-elevated); border-radius: 6px; padding: 6px 10px; font-size: 10px; color: var(--text-secondary); text-align: center; border-left: 3px solid var(--primary); flex-shrink: 0; }
 .yue2-trainer-status.error { border-left-color: var(--danger); color: var(--danger); }
 .yue2-trainer-status.success { border-left-color: var(--success); color: var(--success); }
 .yue2-trainer-preview-section { background: var(--bg-elevated); border-radius: 6px; padding: 8px 10px; flex-shrink: 0; display: flex; flex-direction: column; overflow: hidden; }
 .yue2-trainer-preview-header { font-size: 9px; color: var(--text-secondary); margin-bottom: 6px; flex-shrink: 0; }
 .yue2-trainer-preview-carousel { display: flex; gap: 6px; overflow-x: auto; overflow-y: hidden; align-items: center; padding-bottom: 4px; scrollbar-width: thin; scrollbar-color: var(--border-hover) transparent; }
 .yue2-trainer-preview-carousel::-webkit-scrollbar { height: 4px; }
 .yue2-trainer-preview-carousel::-webkit-scrollbar-track { background: transparent; }
 .yue2-trainer-preview-carousel::-webkit-scrollbar-thumb { background: var(--border-hover); border-radius: 2px; }
 .yue2-trainer-preview-empty { width: 100%; text-align: center; font-size: 10px; color: var(--text-muted); }
 .yue2-trainer-preview-tile { flex-shrink: 0; display: flex; flex-direction: column; align-items: center; gap: 3px; }
 .yue2-trainer-play-btn { width: 40px; height: 40px; border-radius: 50%; border: 2px solid var(--border); background: var(--bg-dark); cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s ease; color: var(--text-secondary); font-size: 14px; }
 .yue2-trainer-play-btn:hover { border-color: var(--primary); color: var(--primary); }
 .yue2-trainer-play-btn.playing { border-color: var(--success); color: var(--success); animation: yue2-trainer-pulse 1.5s infinite; }
 .yue2-trainer-preview-tile .tile-label { font-size: 9px; color: var(--text-muted); text-align: center; white-space: nowrap; }

.yue2-trainer-header button{background:transparent;border:1px solid #3f3f46;border-radius:4px;color:#a1a1aa;padding:3px 7px;font-size:10px;font-family:inherit;cursor:pointer}
.yue2-trainer-chart-plot{flex:1;min-height:0;position:relative}.yue2-trainer-chart-canvas{position:absolute;inset:0}
.yue2-trainer-legend{display:flex;gap:10px;font-size:9px;flex-wrap:wrap}.yue2-trainer-preview-carousel{min-height:62px}.yue2-trainer-preview-tile{min-width:58px}
.yue2-trainer-select{font-size:9px;color:#a1a1aa;background:#0f0f12;border:1px solid #3f3f46;border-radius:4px;padding:3px 6px;cursor:pointer}.yue2-trainer-preview-tile.selected .yue2-trainer-select{border-color:#06b6d4;color:#06b6d4}
.yue2-trainer-play-btn:disabled{opacity:.35;cursor:default}
.yue2-trainer-badge.error{background:#ef4444}.yue2-trainer-badge.complete{background:#22c55e}
`;
document.head.append(css);

function element(tag, parent, text) {
    const node = document.createElement(tag);
    if (text != null) node.textContent = text;
    parent?.append(node);
    return node;
}

async function json(path, body) {
    const response = await api.fetchApi(path, body ? {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)} : {});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Request failed");
    return result;
}

class TrainingPanel {
    constructor(node, caption) {
        this.node = node; this.caption = caption; this.metrics = []; this.job = null; this.activeOperation = null;
        this.createUI();
        this.listener = event => {
            if (String(event.detail.node) !== String(node.id)) return;
            const value = event.detail;
            if (value.job && value.job !== this.job) {
                this.job = value.job;
                this.activeOperation = value.operation;
                if (value.operation === "train") this.load().catch(() => {});
            }
            if (value.type === "complete" || value.type === "error") this.activeOperation = null;
            if (value.type === "status" || value.type === "error") this.setStatus(value.message, value.type === "error" ? "error" : value.operation === "preview" ? "preview" : "running");
            if (value.type === "preview_progress") this.updatePreview?.(value);
            if ((value.type === "complete" || value.type === "error") && this.previewProgress) this.previewProgress.hidden = true;
            if (value.type === "progress") {
                this.updateProgress(value);
                if (value.loss != null) { this.metrics.push(value); this.draw(); }
            }
            if (value.run) { this.node.properties.yue2_run = value.run.replaceAll("\\", "/").split("/").at(-2); }
            if (value.type === "checkpoint") this.load().catch(() => {});
            if (value.type === "complete") { this.setStatus("Complete", "complete"); this.load().catch(() => {}); }
        };
        api.addEventListener("fl_yue2.training", this.listener);
    }
    createUI() {
        const caption = this.caption;
        const pairedPrepare = this.node.comfyClass === "FL_YuE2_PrepareAudioPairs";
        this.root = element("div"); this.root.className = "yue2-training";
        element("h3", this.root, pairedPrepare ? "Prepare audio pairs" : caption ? "Music captions & lyrics" : "YuE2 training studio");
        this.status = element("div", this.root, pairedPrepare ? "Queue to encode aligned source and target recordings." : caption ? "Enter a Google API key above, then queue to send selected recordings to Google." : "Queue to begin. Saved checkpoints remain available after interruption.");
        this.progress = element("progress", this.root); this.progress.max = 1; this.progress.value = 0;
        this.stats = element("small", this.root);
        if (!caption && !pairedPrepare) { this.chart = element("canvas", this.root); this.chart.width = 900; this.chart.height = 220; }
        if (!pairedPrepare) {
            this.refresh = element("button", this.root, caption ? "Load captions" : "Refresh saved run");
            this.refresh.onclick = () => this.load().catch(e => this.status.textContent = e.message);
        }
        this.items = element("div", this.root);
    }
    setStatus(message) { this.status.textContent = message; }
    updateProgress(value) {
        this.progress.value = value.step / value.max_steps;
        this.stats.textContent = `Step ${value.step} / ${value.max_steps}` + (value.loss != null ? ` | loss ${value.loss.toFixed(4)} | LR ${value.lr.toExponential(2)} | peak ${value.peak_gb.toFixed(1)} GB` : "");
    }
    draw() {
        if (!this.chart) return;
        const ctx = this.chart.getContext("2d"), w = this.chart.width, h = this.chart.height;
        ctx.clearRect(0, 0, w, h);
        const series = [["loss", "#55d8c8"], ["artist_validation", "#edb16f"], ["generated_validation", "#c09bfd"]];
        const max = Math.max(0.01, ...this.metrics.flatMap(m => series.map(([key]) => m[key]).filter(Number.isFinite)));
        const last = Math.max(1, ...this.metrics.map(m => m.step));
        series.forEach(([key, color], index) => {
            const values = this.metrics.filter(m => Number.isFinite(m[key]));
            ctx.fillStyle = color; ctx.font = "17px system-ui"; ctx.fillText(key.replaceAll("_", " "), 12 + index * 285, 21);
            ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
            values.forEach((m, i) => { const x = 12 + m.step / last * (w - 24), y = h - 12 - m[key] / max * (h - 48); i ? ctx.lineTo(x,y) : ctx.moveTo(x,y); }); ctx.stroke();
        });
    }
    async load() {
        if (this.caption) {
            const id = this.node.properties.yue2_captions;
            if (!id) return;
            const data = await json(`/fl_yue2/captions/${id}`);
            this.items.replaceChildren();
            for (const song of data.songs) {
                const card = element("article", this.items); element("b", card, song.name);
                const state = element("small", card, "Automatically accepted - editable");
                const style = element("textarea", card); style.value = song.style; style.setAttribute("aria-label", "Style caption");
                const lyrics = element("textarea", card); lyrics.value = song.lyrics; lyrics.style.minHeight = "170px"; lyrics.setAttribute("aria-label", "Full lyrics");
                if (song.uncertainty) element("small", card, song.uncertainty).className = "warning";
                const save = element("button", card, "Save changes");
                save.onclick = async () => {
                    try { await json("/fl_yue2/captions/review", {identifier:id, name:song.name, style:style.value, lyrics:lyrics.value}); state.textContent = "Changes saved"; }
                    catch (e) { state.textContent = e.message; }
                };
            }
        } else {
            const name = this.node.widgets?.find(w => w.name === "output_name")?.value || this.node.properties.yue2_run;
            if (!name) return;
            const request = this.runRequest = (this.runRequest || 0) + 1;
            const run = await json(`/fl_yue2/training/run/${encodeURIComponent(name)}`);
            if (request !== this.runRequest) return;
            this.metrics = run.metrics; this.draw();
            this.updateProgress({...run.metrics.at(-1), step: run.step, max_steps: run.config.steps});
            if (this.activeOperation !== "preview") this.setStatus(`${run.status} - ${run.mode.toUpperCase()} - step ${run.step}`, run.status);
            this.showCheckpoints(run, name);
        }
    }
    destroy() { api.removeEventListener("fl_yue2.training", this.listener); this.root.querySelectorAll("audio").forEach(a => {a.pause(); a.removeAttribute("src");}); this.root.remove(); }
}


class TrainerPanel extends TrainingPanel {
    createUI() {
        this.root = element("div"); this.root.className = "yue2-trainer-widget";
        const section = (name, parent = this.root, tag = "div", text) => {
            const el = element(tag, parent, text); el.className = `yue2-trainer-${name}`; return el;
        };
        const header = section("header"), title = section("title", header, "div", "YuE2 Training");
        this.badge = section("badge idle", title, "span", "Idle");
        this.refresh = element("button", header, "Refresh"); this.refresh.title = "Reload saved run and checkpoint audio";
        this.refresh.onclick = () => this.load().catch(e => this.setStatus(e.message, "error"));
        const content = section("content"), stats = section("stats", content);
        this.values = {};
        for (const key of ["Step", "Loss", "LR"]) {
            const stat = section("stat", stats); section("stat-label", stat, "span", key);
            this.values[key] = section("stat-value", stat, "span", key === "Step" ? "0/0" : "-");
        }
        const progress = section("progress-section", content), row = section("progress-header", progress);
        section("progress-label", row, "span", "Training Progress");
        this.percent = section("progress-value", row, "span", "0%");
        this.progress = section("progress-bar", progress); this.progress.setAttribute("role", "progressbar");
        this.progress.setAttribute("aria-label", "Training progress"); this.progress.setAttribute("aria-valuemin", "0"); this.progress.setAttribute("aria-valuemax", "100");
        this.fill = section("progress-fill", this.progress);
        const chart = section("chart-section", content); section("chart-header", chart, "div", "Loss History");
        const legend = section("legend", chart);
        this.series = this.node.comfyClass === "FL_YuE2_AudioAdapterTrainer"
            ? [["loss", "Training", "#06b6d4"], ["artist_validation", "Paired validation", "#f59e0b"], ["wrong_source_flow", "Wrong source", "#8b5cf6"]]
            : [["loss", "Training", "#06b6d4"], ["artist_validation", "Artist validation", "#f59e0b"], ["generated_validation", "Generated validation", "#8b5cf6"]];
        for (const [, label, color] of this.series) { const el = element("span", legend, label); el.style.color = color; }
        const plot = section("chart-plot", chart); this.chart = section("chart-canvas", plot, "canvas");
        this.status = section("status", content, "div", "Ready to train"); this.status.setAttribute("role", "status");
        const preview = section("preview-section", content); section("preview-header", preview, "div", "Validation Samples");
        this.previewProgress = element("div", preview); this.previewProgress.hidden = true;
        this.previewLabel = element("div", this.previewProgress); this.previewLabel.style.fontSize = "10px";
        this.previewBar = element("progress", this.previewProgress); this.previewBar.style.width = "100%"; this.previewBar.style.accentColor = "#06b6d4";
        this.previewBar.max = 1; this.previewBar.setAttribute("aria-label", "Validation inference progress");
        this.items = section("preview-carousel", preview);
        section("preview-empty", this.items, "div", "Audio appears at each saved checkpoint when render_previews is enabled");
        this.playbackLabel = element("div", preview, "Choose a sample to preview"); this.playbackLabel.style.fontSize = "10px";
        this.seekBar = element("input", preview); this.seekBar.type = "range";
        this.seekBar.min = "0"; this.seekBar.max = "0"; this.seekBar.step = "0.01"; this.seekBar.value = "0"; this.seekBar.disabled = true;
        this.seekBar.style.cssText = "width:100%;margin:6px 0;accent-color:#06b6d4;cursor:pointer";
        this.seekBar.setAttribute("aria-label", "Seek validation sample");
        this.seekBar.onpointerdown = event => event.stopPropagation();
        this.seekBar.onkeydown = event => event.stopPropagation();
        this.seekBar.oninput = () => {
            if (!this.activeAudio || this.seekBar.disabled) return;
            this.activeAudio.currentTime = Math.min(Number(this.seekBar.value), this.activeAudio.duration);
            this.updateSeek();
        };
        this.resizeObserver = new ResizeObserver(() => this.draw()); this.resizeObserver.observe(plot);
    }
    setStatus(message, state = "idle") {
        this.status.textContent = message;
        const active = state === "running" || state === "preview", complete = state === "complete", error = state === "error" || state === "failed";
        this.badge.className = `yue2-trainer-badge ${active ? "training" : complete ? "complete" : error ? "error" : "idle"}`;
        this.badge.textContent = state === "preview" ? "Previewing" : active ? "Training" : complete ? "Complete" : error ? "Error" : state === "cancelled" ? "Cancelled" : state === "selected" ? "Selected" : "Idle";
        this.status.className = `yue2-trainer-status${error ? " error" : complete ? " success" : ""}`;
    }
    updateProgress(value) {
        const percent = Math.min(100, Math.max(0, value.step / value.max_steps * 100));
        this.values.Step.textContent = `${value.step}/${value.max_steps}`;
        this.values.Loss.textContent = Number.isFinite(value.loss) ? value.loss.toFixed(6) : "-";
        this.values.LR.textContent = Number.isFinite(value.lr) ? value.lr.toExponential(2) : "-";
        this.fill.style.width = `${percent}%`; this.percent.textContent = `${percent.toFixed(1)}%`;
        this.progress.setAttribute("aria-valuenow", String(percent));
        if (this.activeOperation !== "preview") this.setStatus(`Training step ${value.step} of ${value.max_steps}`, "running");
    }
    draw() {
        const w = this.chart.clientWidth, h = this.chart.clientHeight;
        if (w < 80 || h < 40) return;
        const ratio = window.devicePixelRatio || 1;
        this.chart.width = Math.round(w * ratio); this.chart.height = Math.round(h * ratio);
        const ctx = this.chart.getContext("2d"); ctx.scale(ratio, ratio);
        ctx.fillStyle = "#0f0f12"; ctx.fillRect(0, 0, w, h);
        const points = this.metrics.filter(m => Number.isFinite(m.loss));
        ctx.font = "10px Inter, sans-serif";
        if (!points.length) { ctx.fillStyle = "#71717a"; ctx.textAlign = "center"; ctx.fillText("Waiting for training data...", w/2, h/2); return; }
        const values = points.flatMap(m => this.series.map(([key]) => m[key]).filter(Number.isFinite));
        const min = Math.min(...values), max = Math.max(...values), range = max - min || 1;
        const first = points[0].step, last = points.at(-1).step;
        const left = 48, top = 20, width = w - 62, height = h - 44;
        const x = m => left + (m.step - first) / (last - first || 1) * width;
        const y = value => top + height - (value - min) / range * height;
        ctx.font = "9px Inter, sans-serif";
        for (let i = 0; i <= 4; i++) {
            const gy = top + height * i/4;
            ctx.strokeStyle = "#27272a"; ctx.lineWidth = .5; ctx.beginPath(); ctx.moveTo(left, gy); ctx.lineTo(w-14, gy); ctx.stroke();
            ctx.fillStyle = "#71717a"; ctx.textAlign = "right"; ctx.fillText((max-range*i/4).toFixed(4), left-5, gy+3);
        }
        ctx.textAlign = "left"; ctx.fillText(String(first), left, h-7); ctx.textAlign = "right"; ctx.fillText(String(last), w-14, h-7);
        for (const [key,,color] of this.series) {
            const line = points.filter(m => Number.isFinite(m[key]));
            ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.beginPath();
            line.forEach((m,i) => i ? ctx.lineTo(x(m), y(m[key])) : ctx.moveTo(x(m), y(m[key]))); ctx.stroke();
            if (line.length === 1) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x(line[0]), y(line[0][key]), 2, 0, 2*Math.PI); ctx.fill(); }
            if (key === "loss") {
                const gradient = ctx.createLinearGradient(0,top,0,top+height); gradient.addColorStop(0,"rgba(6,182,212,.3)"); gradient.addColorStop(1,"rgba(6,182,212,0)");
                ctx.lineTo(x(line.at(-1)),top+height); ctx.lineTo(x(line[0]),top+height); ctx.closePath(); ctx.fillStyle=gradient; ctx.fill();
            }
        }
    }
    updatePreview(value) {
        this.previewProgress.hidden = false;
        const labels = {loading: "Loading models", tokens: "Generating music", synthesis: "Synthesizing audio", decode: "Decoding audio", saving: "Saving sample", complete: "Sample ready"};
        const detail = value.total > 0 ? value.phase === "tokens" ? ` - ${(value.done / 25).toFixed(1)} / ${(value.total / 25).toFixed(1)}s maximum` : ` - ${value.done}/${value.total}` : "";
        this.previewLabel.textContent = `${value.step === 0 ? "Baseline - Step 0" : `Step ${value.step}`} - ${labels[value.phase] || value.phase}${detail}`;
        if (value.total > 0) this.previewBar.value = value.done / value.total;
        else this.previewBar.removeAttribute("value");
        this.setStatus(this.previewLabel.textContent, "preview");
    }
    updateSeek() {
        const audio = this.activeAudio;
        const duration = audio && Number.isFinite(audio.duration) ? audio.duration : 0;
        const current = audio ? audio.currentTime : 0;
        const clock = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
        this.seekBar.disabled = duration <= 0;
        this.seekBar.max = String(duration); this.seekBar.value = String(current);
        this.playbackLabel.textContent = audio ? `${this.activeSample} - ${clock(current)} / ${duration > 0 ? clock(duration) : "Loading..."}` : "Choose a sample to preview";
        this.seekBar.setAttribute("aria-valuetext", audio ? `${clock(current)} of ${clock(duration)}` : "No sample selected");
    }
    showCheckpoints(run, name) {
        const samples = [...(run.references || []), ...(run.baseline ? [run.baseline] : []), ...run.checkpoints];
        const existing = new Map([...this.items.children].map(tile => [tile.dataset.step, tile]));
        for (const tile of existing.values()) {
            const checkpoint = samples.find(c => String(c.step) === tile.dataset.step);
            if (tile.dataset.run !== name || !checkpoint || tile.dataset.preview !== (checkpoint.preview || "")) {
                const audio = tile.querySelector("audio");
                if (audio && audio === this.activeAudio) { this.activeAudio = null; this.updateSeek(); }
                if (audio) { audio.pause(); audio.removeAttribute("src"); audio.load(); }
                tile.remove(); existing.delete(tile.dataset.step);
            }
        }
        if (!samples.length) { const empty = element("div",this.items,"Audio appears at each saved checkpoint when render_previews is enabled"); empty.className="yue2-trainer-preview-empty"; }
        for (const checkpoint of samples) {
            if (existing.has(String(checkpoint.step))) continue;
            const tile = element("div",this.items); tile.className="yue2-trainer-preview-tile"; tile.dataset.step=checkpoint.step;
            tile.dataset.run=name; tile.dataset.preview=checkpoint.preview || "";
            const play = element("button",tile,"\u25b6"); play.className="yue2-trainer-play-btn"; play.setAttribute("aria-label",`Play checkpoint ${checkpoint.step}`);
            const label = element("div",tile,checkpoint.label || (checkpoint.step === 0 ? "Baseline - Step 0" : `S${checkpoint.step}`)); label.className="tile-label";
            const select = element("button",tile,"Use"); select.className="yue2-trainer-select";
            select.hidden = checkpoint.step <= 0;
            select.setAttribute("aria-label",`Use checkpoint ${checkpoint.step}`);
            select.onclick=()=>{
                this.node.widgets.find(w=>w.name==="selected_step").value=checkpoint.step;
                this.node.widgets.find(w=>w.name==="action").value="use_saved";
                this.markSelected(); this.node.graph?.change(); this.node.setDirtyCanvas(true,true);
                this.setStatus(`Step ${checkpoint.step} selected. Queue to use it without retraining.`, "selected");
            };
            play.disabled = !checkpoint.preview;
            play.title = checkpoint.preview ? `Play step ${checkpoint.step}` : "Audio is not ready yet. Enable render_previews to render a sample at each saved checkpoint before training resumes.";
            if (!checkpoint.preview) continue;
            const audio = element("audio",tile); audio.preload="none";
            audio.src=api.apiURL(`/fl_yue2/training/audio/${encodeURIComponent(name)}/${encodeURIComponent(checkpoint.preview)}`);
            const syncSeek = () => { if (this.activeAudio === audio) this.updateSeek(); };
            audio.ontimeupdate = syncSeek; audio.onloadedmetadata = syncSeek; audio.ondurationchange = syncSeek;
            const stopped=()=>{play.textContent="\u25b6";play.classList.remove("playing");play.setAttribute("aria-label",`Play checkpoint ${checkpoint.step}`);};
            audio.onpause=stopped;audio.onended=stopped;
            play.onclick=async()=>{
                this.activeAudio = audio; this.activeSample = checkpoint.label || (checkpoint.step === 0 ? "Baseline - Step 0" : `Step ${checkpoint.step}`);
                this.updateSeek();
                if (!audio.paused) {audio.pause();return;}
                this.items.querySelectorAll("audio").forEach(other=>{if(other!==audio){other.pause();other.currentTime=0;}});
                try {await audio.play();play.textContent="\u2161";play.classList.add("playing");play.setAttribute("aria-label",`Pause checkpoint ${checkpoint.step}`);}
                catch(e){this.setStatus(`Cannot play checkpoint: ${e.message}`,"error");}
            };
        }
        samples.forEach((checkpoint, index) => {
            const tile = this.items.querySelector(`[data-step="${checkpoint.step}"]`);
            if (this.items.children[index] !== tile) this.items.insertBefore(tile, this.items.children[index] || null);
        });
        this.markSelected();
    }
    markSelected() {
        const step=Number(this.node.widgets.find(w=>w.name==="selected_step").value) || Number(this.items.lastElementChild?.dataset.step);
        for(const tile of this.items.querySelectorAll("[data-step]")) {
            const selected=Number(tile.dataset.step)>0 && Number(tile.dataset.step)===step;
            tile.classList.toggle("selected",selected);
            const button=tile.querySelector(".yue2-trainer-select");button.textContent=selected?"Selected":"Use";button.setAttribute("aria-pressed",String(selected));
        }
    }
    destroy() { this.resizeObserver.disconnect(); super.destroy(); }
}

app.registerExtension({
    name:"FL.YuE2.Training",
    beforeRegisterNodeDef(type, data) {
        if (!["FL_YuE2_GeminiMusicCaptioner", "FL_YuE2_LoRATrainer", "FL_YuE2_PrepareDataset", "FL_YuE2_AudioAdapterTrainer", "FL_YuE2_PrepareAudioPairs"].includes(data.name)) return;
        const created = type.prototype.onNodeCreated, executed = type.prototype.onExecuted, removed = type.prototype.onRemoved, configured = type.prototype.onConfigure;
        type.prototype.onNodeCreated = function() {
            created?.apply(this, arguments); this.properties ||= {};
            const trainer = ["FL_YuE2_LoRATrainer", "FL_YuE2_AudioAdapterTrainer"].includes(data.name);
            const pairedPrepare = data.name === "FL_YuE2_PrepareAudioPairs";
            this.yue2Training = trainer ? new TrainerPanel(this, false) : new TrainingPanel(this, data.name === "FL_YuE2_GeminiMusicCaptioner");
            const widget = this.addDOMWidget("yue2_training", "custom", this.yue2Training.root, {serialize:false, hideOnZoom:false, getMinHeight: () => pairedPrepare ? 150 : trainer ? 400 : 450});
            if (!trainer) widget.computeSize = () => pairedPrepare ? [360, 150] : [520, 450];
            this.setSize([Math.max(trainer || pairedPrepare ? 400 : 550,this.size[0]), Math.max(pairedPrepare ? 270 : trainer ? 500 : 760,this.size[1])]);
        };
        type.prototype.onExecuted = function(message) { executed?.apply(this,arguments); if (message.run) this.properties.yue2_run = message.run[0]; if (message.captions) this.properties.yue2_captions = message.captions[0]; this.yue2Training?.load().catch(e => this.yue2Training.status.textContent = e.message); };
        type.prototype.onConfigure = function() { configured?.apply(this,arguments); this.yue2Training?.load().catch(() => {}); };
        type.prototype.onRemoved = function() { this.yue2Training?.destroy(); removed?.apply(this,arguments); };
    }
});
