import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PRE_MODES = {
    ASK: "Ask Approval in UI (Wait After 768p Draft)",
    AUTO_2X: "Pre-Approve 2x Upscale (Auto 2K — No Pause)",
    AUTO_4X: "Pre-Approve 4x Upscale (Auto 4K — No Pause)",
    AUTO_CUSTOM: "Pre-Approve Custom Scale (Auto — No Pause)",
    STOP: "Draft Only (Stop After 768p — No Upscale)",
};

const GATE_MODES = {
    ASK: "Ask Approval in UI (Wait Indefinitely - No Timer)",
    AUTO: "Auto-Upscale Immediately (No Pause)",
    STOP: "Stop After 768p Draft Only (Do Not Upscale)",
};

const GATE_FACTORS = {
    X2: "2x Upscale (1440p / 2K)",
    X4: "4x Upscale (2880p / 4K)",
};

let floatingBanner = null;
let isSyncingNodes = false;

function findNodesByClass(className) {
    const nodes = app.graph?._nodes || [];
    return nodes.filter((n) => n && (n.type === className || n.comfyClass === className));
}

function getWidget(node, name) {
    return node?.widgets?.find((w) => w.name === name) || null;
}

function formatScale(val) {
    const num = Number(val);
    if (!Number.isFinite(num)) return "2";
    return Number.isInteger(num) ? String(num) : String(Number(num.toFixed(2)));
}

function sendApprovalDecision(nodeId, decision, extraPayload = {}) {
    return fetch("/h3_approval/decision", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            node_id: String(nodeId || "latest"),
            decision,
            ...extraPayload,
        }),
    })
        .then((r) => r.json())
        .then(() => {
            hideFloatingBanner();
        })
        .catch((err) => {
            console.error("[H3UpscaleApprovalGate] Failed to send decision:", err);
        });
}

async function resolveIfGateWaiting(preMode, customScale) {
    let decision = null;
    const extra = {};
    if (preMode === PRE_MODES.AUTO_2X) {
        decision = "approve_2x";
    } else if (preMode === PRE_MODES.AUTO_4X) {
        decision = "approve_4x";
    } else if (preMode === PRE_MODES.AUTO_CUSTOM) {
        decision = "approve_custom";
        extra.scale = Number(customScale) || 2.0;
    } else if (preMode === PRE_MODES.STOP) {
        decision = "stop";
    }
    if (!decision) return;

    try {
        const resp = await fetch("/h3_approval/status");
        if (!resp.ok) return;
        const data = await resp.json();
        const waitingEntry = Object.entries(data.states || {}).find(
            ([, v]) => v && v.status === "waiting"
        );
        if (waitingEntry) {
            await sendApprovalDecision(waitingEntry[0], decision, extra);
        }
    } catch (e) {
        // Offline or idle server — preset still stays saved on the workflow node
    }
}

function hideFloatingBanner() {
    if (floatingBanner && floatingBanner.parentNode) {
        floatingBanner.parentNode.removeChild(floatingBanner);
    }
    floatingBanner = null;
}

function showFloatingBanner(nodeId, upscaleFactor, scaleFloat = 2.0) {
    hideFloatingBanner();

    const bar = document.createElement("div");
    bar.id = "h3-approval-floating-banner";
    bar.style.cssText = [
        "position: fixed",
        "top: 56px",
        "left: 50%",
        "transform: translateX(-50%)",
        "z-index: 99999",
        "background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
        "border: 2px solid #22c55e",
        "border-radius: 12px",
        "padding: 14px 22px",
        "box-shadow: 0 12px 32px rgba(0,0,0,0.75)",
        "color: #f8fafc",
        "font-family: system-ui, -apple-system, sans-serif",
        "display: flex",
        "flex-direction: column",
        "gap: 10px",
        "min-width: 520px",
        "max-width: 92vw",
    ].join(";");

    const titleRow = document.createElement("div");
    titleRow.style.cssText = "display: flex; align-items: center; justify-content: space-between; gap: 12px;";
    titleRow.innerHTML = `
        <div style="font-size: 15px; font-weight: 700; color: #4ade80;">
            🎬 Stage 1 (768p Draft Video) is Ready! (Paused — No Timer)
        </div>
        <div style="font-size: 12px; color: #94a3b8;">
            Inspect the 768p output first, then choose below:
        </div>
    `;

    const btnRow = document.createElement("div");
    btnRow.style.cssText = "display: flex; align-items: center; gap: 10px; flex-wrap: wrap;";

    const makeBtn = (label, bg, onClick) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.style.cssText = [
            `background: ${bg}`,
            "color: #ffffff",
            "border: none",
            "border-radius: 8px",
            "padding: 9px 16px",
            "font-size: 13px",
            "font-weight: 700",
            "cursor: pointer",
            "transition: filter 0.15s ease",
            "flex: 1",
            "white-space: nowrap",
        ].join(";");
        b.onmouseenter = () => (b.style.filter = "brightness(1.15)");
        b.onmouseleave = () => (b.style.filter = "none");
        b.onclick = onClick;
        return b;
    };

    btnRow.appendChild(
        makeBtn("✅ Approve & Upscale (2x / 2K)", "#16a34a", () =>
            sendApprovalDecision(nodeId, "approve_2x")
        )
    );
    btnRow.appendChild(
        makeBtn("🚀 Approve & Upscale (4x / 4K)", "#2563eb", () =>
            sendApprovalDecision(nodeId, "approve_4x")
        )
    );

    const parsedScale = Number(scaleFloat);
    if (Number.isFinite(parsedScale) && Math.abs(parsedScale - 2.0) > 0.05 && Math.abs(parsedScale - 4.0) > 0.05) {
        btnRow.appendChild(
            makeBtn(`🎛️ Approve (${formatScale(parsedScale)}x)`, "#7c3aed", () =>
                sendApprovalDecision(nodeId, "approve_custom", { scale: parsedScale })
            )
        );
    }

    btnRow.appendChild(
        makeBtn("🛑 Stop Here (Keep 768p Only)", "#dc2626", () =>
            sendApprovalDecision(nodeId, "stop")
        )
    );

    bar.appendChild(titleRow);
    bar.appendChild(btnRow);
    document.body.appendChild(bar);
    floatingBanner = bar;
}

const PRE_BUTTON_DEFS = [
    {
        mode: PRE_MODES.ASK,
        offLabel: "⚪ Ask Approval (Pause After 768p Draft)",
        onLabel: () => "🟢 ON · Ask Approval (Wait at 768p)",
    },
    {
        mode: PRE_MODES.AUTO_2X,
        offLabel: "⚪ Pre-Approve 2x Upscale (Auto 2K — No Pause)",
        onLabel: () => "🟢 ON · Pre-Approved 2x (Auto 2K)",
    },
    {
        mode: PRE_MODES.AUTO_4X,
        offLabel: "⚪ Pre-Approve 4x Upscale (Auto 4K — No Pause)",
        onLabel: () => "🟢 ON · Pre-Approved 4x (Auto 4K)",
    },
    {
        mode: PRE_MODES.AUTO_CUSTOM,
        offLabel: "⚪ Pre-Approve Custom Scale (Auto — No Pause)",
        onLabel: (scale) => `🟢 ON · Pre-Approved ${formatScale(scale)}x Custom`,
    },
    {
        mode: PRE_MODES.STOP,
        offLabel: "⚪ Draft Only (Stop After 768p — No Upscale)",
        onLabel: () => "🟢 ON · Draft Only (Stop at 768p)",
    },
];

function applyPreNodeColors(node, mode) {
    if (mode === PRE_MODES.STOP) {
        node.color = "#7f1d1d";
        node.bgcolor = "#450a0a";
    } else if (mode === PRE_MODES.ASK) {
        node.color = "#78350f";
        node.bgcolor = "#451a03";
    } else {
        node.color = "#14532d";
        node.bgcolor = "#052e16";
    }
}

function syncGatesFromPreNode(mode, customScale) {
    const gates = findNodesByClass("H3UpscaleApprovalGate");
    if (!gates.length) return;

    let gateMode = GATE_MODES.ASK;
    let gateFactor = Number(customScale) >= 3.5 ? GATE_FACTORS.X4 : GATE_FACTORS.X2;
    if (mode === PRE_MODES.AUTO_2X) {
        gateMode = GATE_MODES.AUTO;
        gateFactor = GATE_FACTORS.X2;
    } else if (mode === PRE_MODES.AUTO_4X) {
        gateMode = GATE_MODES.AUTO;
        gateFactor = GATE_FACTORS.X4;
    } else if (mode === PRE_MODES.AUTO_CUSTOM) {
        gateMode = GATE_MODES.AUTO;
    } else if (mode === PRE_MODES.STOP) {
        gateMode = GATE_MODES.STOP;
    }

    isSyncingNodes = true;
    try {
        for (const g of gates) {
            const wMode = getWidget(g, "approval_mode");
            const wFactor = getWidget(g, "upscale_factor");
            if (wMode) wMode.value = gateMode;
            if (wFactor) wFactor.value = gateFactor;
            g.setDirtyCanvas?.(true, true);
        }
    } finally {
        isSyncingNodes = false;
    }
}

function syncPreApprovalUI(node, propagateToGates = false) {
    if (!node) return;
    const modeWidget = getWidget(node, "upscale_mode");
    const scaleWidget = getWidget(node, "custom_scale");
    const activeMode = modeWidget?.value || PRE_MODES.ASK;
    const customScale = scaleWidget?.value ?? 2.0;

    if (Array.isArray(node._h3ModeButtons)) {
        for (const entry of node._h3ModeButtons) {
            const isOn = entry.spec.mode === activeMode;
            entry.widget.name = isOn ? entry.spec.onLabel(customScale) : entry.spec.offLabel;
        }
    }
    applyPreNodeColors(node, activeMode);

    if (propagateToGates && !isSyncingNodes) {
        syncGatesFromPreNode(activeMode, customScale);
    }

    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function syncPreNodesFromGate(gateNode) {
    if (isSyncingNodes) return;
    const preNodes = findNodesByClass("H3UpscalePreApproval");
    if (!preNodes.length) return;

    const gateMode = getWidget(gateNode, "approval_mode")?.value || GATE_MODES.ASK;
    const gateFactor = getWidget(gateNode, "upscale_factor")?.value || GATE_FACTORS.X2;

    let targetMode = PRE_MODES.ASK;
    if (gateMode === GATE_MODES.STOP) {
        targetMode = PRE_MODES.STOP;
    } else if (gateMode === GATE_MODES.AUTO) {
        targetMode = String(gateFactor).startsWith("4x") ? PRE_MODES.AUTO_4X : PRE_MODES.AUTO_2X;
    }

    isSyncingNodes = true;
    try {
        for (const pre of preNodes) {
            const wMode = getWidget(pre, "upscale_mode");
            if (wMode && wMode.value !== targetMode) {
                wMode.value = targetMode;
            }
            syncPreApprovalUI(pre, false);
        }
    } finally {
        isSyncingNodes = false;
    }
}

function hookWidgetCallback(widget, onChange) {
    if (!widget || widget._h3Hooked) return;
    const prev = widget.callback;
    widget.callback = function (...args) {
        const res = prev ? prev.apply(this, args) : undefined;
        onChange();
        return res;
    };
    widget._h3Hooked = true;
}

app.registerExtension({
    name: "MiniMaxH3.UpscaleApprovalGate",
    async setup() {
        api.addEventListener("h3_approval_waiting", (event) => {
            const detail = event.detail || {};
            showFloatingBanner(
                detail.node_id || "latest",
                detail.upscale_factor || "2x",
                detail.scale_float ?? 2.0
            );
        });

        api.addEventListener("h3_approval_resolved", () => {
            hideFloatingBanner();
        });

        api.addEventListener("execution_interrupted", () => {
            hideFloatingBanner();
        });

        api.addEventListener("execution_error", () => {
            hideFloatingBanner();
        });

        // Fallback poll every 1.5s so even if a reverse proxy strips custom WS events,
        // the floating approval bar still appears as soon as Stage 1 finishes!
        setInterval(async () => {
            try {
                const resp = await fetch("/h3_approval/status");
                if (!resp.ok) return;
                const data = await resp.json();
                const states = data.states || {};
                const waitingEntry = Object.entries(states).find(
                    ([, v]) => v && v.status === "waiting"
                );
                if (waitingEntry) {
                    if (!floatingBanner) {
                        showFloatingBanner(waitingEntry[0], "2x", waitingEntry[1]?.scale ?? 2.0);
                    }
                } else if (floatingBanner) {
                    hideFloatingBanner();
                }
            } catch (e) {
                // ignore network poll errors
            }
        }, 1500);
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "H3UpscalePreApproval") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const modeWidget = getWidget(this, "upscale_mode");
                const scaleWidget = getWidget(this, "custom_scale");

                hookWidgetCallback(modeWidget, () => syncPreApprovalUI(this, true));
                hookWidgetCallback(scaleWidget, () => syncPreApprovalUI(this, true));

                this._h3ModeButtons = PRE_BUTTON_DEFS.map((spec) => {
                    const widget = this.addWidget("button", spec.offLabel, null, () => {
                        if (modeWidget) {
                            modeWidget.value = spec.mode;
                            modeWidget.callback?.(spec.mode);
                        }
                        syncPreApprovalUI(this, true);
                        resolveIfGateWaiting(spec.mode, scaleWidget?.value ?? 2.0);
                    });
                    return { spec, widget };
                });

                syncPreApprovalUI(this, false);
                return r;
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                setTimeout(() => syncPreApprovalUI(this, false), 0);
                return r;
            };
            return;
        }

        if (nodeData.name === "ReCaVSRLoader") {
            const healLoader = (node) => {
                const w = getWidget(node, "transformer");
                if (!w) return;
                const bad = new Set(["flashdecoder.safetensors", "prompt.safetensors", "vae.safetensors"]);
                if (bad.has(String(w.value || ""))) {
                    const opts = Array.isArray(w.options?.values) ? w.options.values : [];
                    w.value = opts.includes("transformer_int8_convrot.safetensors")
                        ? "transformer_int8_convrot.safetensors"
                        : opts.find((v) => !bad.has(v)) || "transformer_int8_convrot.safetensors";
                    node.setDirtyCanvas?.(true, true);
                }
            };
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                healLoader(this);
                return r;
            };
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                setTimeout(() => healLoader(this), 0);
                return r;
            };
            return;
        }

        if (nodeData.name !== "H3UpscaleApprovalGate") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            hookWidgetCallback(getWidget(this, "approval_mode"), () => syncPreNodesFromGate(this));
            hookWidgetCallback(getWidget(this, "upscale_factor"), () => syncPreNodesFromGate(this));

            this.addWidget("button", "✅ Approve & Upscale 2x (Run Stage 2)", null, () => {
                sendApprovalDecision(this.id, "approve_2x");
            });
            this.addWidget("button", "🚀 Approve & Upscale 4x (Run Stage 2)", null, () => {
                sendApprovalDecision(this.id, "approve_4x");
            });
            this.addWidget("button", "🛑 Stop Here (Keep 768p Draft Only)", null, () => {
                sendApprovalDecision(this.id, "stop");
            });

            this.color = "#14532d";
            this.bgcolor = "#052e16";
            return r;
        };
    },
});
