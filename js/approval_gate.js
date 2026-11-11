import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let floatingBanner = null;

function sendApprovalDecision(nodeId, decision) {
    fetch("/h3_approval/decision", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_id: String(nodeId || "latest"), decision }),
    })
        .then((r) => r.json())
        .then(() => {
            hideFloatingBanner();
        })
        .catch((err) => {
            console.error("[H3UpscaleApprovalGate] Failed to send decision:", err);
        });
}

function hideFloatingBanner() {
    if (floatingBanner && floatingBanner.parentNode) {
        floatingBanner.parentNode.removeChild(floatingBanner);
    }
    floatingBanner = null;
}

function showFloatingBanner(nodeId, upscaleFactor) {
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

    const makeBtn = (label, bg, hoverBg, onClick) => {
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
        makeBtn("✅ Approve & Upscale (2x / 2K)", "#16a34a", "#15803d", () =>
            sendApprovalDecision(nodeId, "approve_2x")
        )
    );
    btnRow.appendChild(
        makeBtn("🚀 Approve & Upscale (4x / 4K)", "#2563eb", "#1d4ed8", () =>
            sendApprovalDecision(nodeId, "approve_4x")
        )
    );
    btnRow.appendChild(
        makeBtn("🛑 Stop Here (Keep 768p Only)", "#dc2626", "#b91c1c", () =>
            sendApprovalDecision(nodeId, "stop")
        )
    );

    bar.appendChild(titleRow);
    bar.appendChild(btnRow);
    document.body.appendChild(bar);
    floatingBanner = bar;
}

app.registerExtension({
    name: "MiniMaxH3.UpscaleApprovalGate",
    async setup() {
        api.addEventListener("h3_approval_waiting", (event) => {
            const detail = event.detail || {};
            showFloatingBanner(detail.node_id || "latest", detail.upscale_factor || "2x");
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
                    ([k, v]) => v && v.status === "waiting"
                );
                if (waitingEntry) {
                    if (!floatingBanner) {
                        showFloatingBanner(waitingEntry[0], "2x");
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
        if (nodeData.name !== "H3UpscaleApprovalGate") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

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
