import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FL.YuE2",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData.name.startsWith("FL_YuE2_")) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            created?.apply(this, arguments);
            this.color = "#16727c";
            this.bgcolor = "#4F0074";
            if (!["FL_YuE2_Plan", "FL_YuE2_Render"].includes(nodeData.name)) return;
            const output = document.createElement("textarea");
            output.readOnly = true;
            output.placeholder = nodeData.name === "FL_YuE2_Plan" ? "Your generated score will appear here after composing." : "Generation status";
            output.setAttribute("aria-label", "YuE2 generation result");
            Object.assign(output.style, { width: "100%", height: "100%", resize: "none", background: "#29003d", color: "#f2e6f7", border: "1px solid #16727c", borderRadius: "6px", padding: "8px", boxSizing: "border-box", fontFamily: "monospace" });
            const widget = this.addDOMWidget("yue2_result", "customtext", output, { serialize: false });
            widget.computeSize = () => [this.size[0], nodeData.name === "FL_YuE2_Plan" ? 150 : 55];
            this.yue2Result = output;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            executed?.apply(this, arguments);
            if (this.yue2Result && message.text) this.yue2Result.value = message.text.join("\n");
        };
    },
});
