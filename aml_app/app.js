// SHIELD AML - Frontend Logic & Real-time Integration

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const statTotalTx = document.getElementById('stat-total-tx');
    const statTier1Count = document.getElementById('stat-tier1-count');
    const statTier1Rate = document.getElementById('stat-tier1-rate');
    const statTier2Count = document.getElementById('stat-tier2-count');
    const statTier2Rate = document.getElementById('stat-tier2-rate');
    const statTier3Count = document.getElementById('stat-tier3-count');
    const statTier3Rate = document.getElementById('stat-tier3-rate');

    const txStreamBody = document.getElementById('tx-stream-body');
    const btnSimulateBatch = document.getElementById('btn-simulate-batch');
    const btnToggleStream = document.getElementById('btn-toggle-stream');
    const evalForm = document.getElementById('eval-form');
    const auditLogContainer = document.getElementById('audit-log-container');

    // Modal Elements
    const reviewModal = document.getElementById('review-modal');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const modalCancelBtn = document.getElementById('modal-cancel-btn');
    const modalSubmitBtn = document.getElementById('modal-submit-btn');
    const modalTxHash = document.getElementById('modal-tx-hash');
    const modalAmount = document.getElementById('modal-amount');
    const modalRisk = document.getElementById('modal-risk');
    const modalUnc = document.getElementById('modal-unc');
    const modalAction = document.getElementById('modal-action');
    const modalDecisionSelect = document.getElementById('modal-decision-select');
    const modalNotes = document.getElementById('modal-notes');

    // State Variables
    let isAutoStreaming = false;
    let autoStreamTimer = null;
    let activeReviewTxId = null;
    let localHistory = [];

    // Helper: Add log entry to Engine Audit Log UI
    function addAuditLog(message, type = 'info') {
        const time = new Date().toLocaleTimeString();
        const entry = document.createElement('div');
        entry.className = `log-entry log-${type}`;
        entry.textContent = `[${time}] ${message}`;
        auditLogContainer.prepend(entry);
        
        while (auditLogContainer.children.length > 50) {
            auditLogContainer.removeChild(auditLogContainer.lastChild);
        }
    }

    // Render Tier Badge HTML based on server status_code / action
    function getActionBadgeHtml(tx) {
        const statusCode = tx.status_code || '';
        const action = tx.action || '';

        if (statusCode.includes('MANUAL') || action.includes('MANUAL')) {
            if (statusCode.includes('APPROVED') || action.includes('APPROVED')) {
                return `<span class="badge badge-approve">OVERRIDDEN: APPROVED</span>`;
            } else {
                return `<span class="badge badge-decline">OVERRIDDEN: DECLINED</span>`;
            }
        }
        
        if (statusCode === 'DECLINED_AND_FREEZE' || tx.tier === 1) {
            return `<span class="badge badge-decline">TIER 1: DECLINE & FREEZE</span>`;
        } else if (statusCode === 'ESCALATE_MANUAL_REVIEW' || tx.tier === 2) {
            return `<span class="badge badge-review">TIER 2: MANUAL REVIEW</span>`;
        } else {
            return `<span class="badge badge-approve">TIER 3: INSTANT APPROVE</span>`;
        }
    }

    // Render Risk Score Pill
    function getRiskScoreHtml(score, uncertainty) {
        let textClass = 'text-green';
        if (score >= 0.90) textClass = 'text-red';
        else if (score >= 0.70 || uncertainty > 0.04) textClass = 'text-orange';

        return `
            <div class="risk-score-pill">
                <span class="score-num ${textClass}">${score.toFixed(4)}</span>
                <span class="uncertainty-tag">&plusmn;${uncertainty.toFixed(4)}</span>
            </div>
        `;
    }

    // Fetch and Update Dashboard Stats
    async function fetchStats() {
        try {
            const res = await fetch('/api/stats');
            if (!res.ok) return;
            const data = await res.json();

            statTotalTx.textContent = data.total_nodes.toLocaleString();
            
            if (data.tier_breakdown) {
                statTier1Count.textContent = data.tier_breakdown.declined.count.toLocaleString();
                statTier1Rate.textContent = `${data.tier_breakdown.declined.pct}% illicit intercepted`;

                statTier2Count.textContent = data.tier_breakdown.manual_review.count.toLocaleString();
                statTier2Rate.textContent = `${data.tier_breakdown.manual_review.pct}% high uncertainty`;

                statTier3Count.textContent = data.tier_breakdown.approved.count.toLocaleString();
                statTier3Rate.textContent = `${data.tier_breakdown.approved.pct}% clean pass-through`;
            }
        } catch (err) {
            console.error('Error fetching stats:', err);
        }
    }

    // Render Transactions Table
    function renderTxTable(transactions) {
        txStreamBody.innerHTML = '';
        if (!transactions || transactions.length === 0) {
            txStreamBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: #718096; padding: 24px;">No transactions in live stream. Click "Simulate Incoming Tx" to populate.</td></tr>`;
            return;
        }

        transactions.forEach(tx => {
            const row = document.createElement('tr');
            const isDecline = (tx.status_code && tx.status_code.includes('DECLINE')) || (tx.action && tx.action.includes('DECLINE'));
            row.className = isDecline ? 'row-highlight-decline' : '';

            const btcVal = tx.btc_volume || tx.btc || 0;
            const usdVal = Math.round(btcVal * 63500);

            row.innerHTML = `
                <td>
                    <div style="font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;" class="text-cyan">${tx.tx_id}</div>
                    <div style="font-size: 0.75rem; color: #718096;">${tx.timestamp}</div>
                </td>
                <td>
                    <div style="font-weight: 600;">${btcVal.toFixed(2)} BTC</div>
                    <div style="font-size: 0.75rem; color: #718096;">$${usdVal.toLocaleString()}</div>
                </td>
                <td>
                    <div>In: <strong>${tx.inputs || 1}</strong> | Out: <strong>${tx.outputs || 1}</strong></div>
                    <div style="font-size: 0.75rem; color: #a0aec0;">${tx.gt_category || 'Graph Flow Node'}</div>
                </td>
                <td>
                    ${getRiskScoreHtml(tx.probability, tx.uncertainty)}
                </td>
                <td>
                    ${getActionBadgeHtml(tx)}
                </td>
                <td>
                    <button class="btn btn-sm btn-secondary btn-review-tx" data-txid="${tx.tx_id}">
                        Inspect / Override
                    </button>
                </td>
            `;

            txStreamBody.appendChild(row);
        });

        // Attach event listeners to "Inspect / Override" buttons
        document.querySelectorAll('.btn-review-tx').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const txId = e.currentTarget.getAttribute('data-txid');
                const targetTx = localHistory.find(t => t.tx_id === txId);
                if (targetTx) openReviewModal(targetTx);
            });
        });
    }

    // Fetch Recent Transaction History
    async function fetchHistory() {
        try {
            const res = await fetch('/api/history');
            if (!res.ok) return;
            const data = await res.json();
            localHistory = data.history || [];
            renderTxTable(localHistory);
        } catch (err) {
            console.error('Error fetching history:', err);
        }
    }

    // Trigger Single Simulation call
    async function triggerSimulateSingle() {
        try {
            const res = await fetch('/api/simulate');
            if (res.ok) {
                const tx = await res.json();
                addAuditLog(`[STREAM] Incoming ${tx.tx_id} -> Prob: ${tx.probability.toFixed(4)} (${tx.action})`, tx.probability >= 0.90 ? 'decline' : 'info');
                await fetchHistory();
            }
        } catch (err) {
            addAuditLog(`[ERROR] Simulation call failed: ${err.message}`, 'error');
        }
    }

    // Trigger Batch Simulation (5 times)
    async function triggerSimulateBatch(count = 3) {
        btnSimulateBatch.disabled = true;
        btnSimulateBatch.textContent = 'Simulating...';

        for (let i = 0; i < count; i++) {
            await triggerSimulateSingle();
        }

        btnSimulateBatch.disabled = false;
        btnSimulateBatch.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            Simulate Incoming Tx
        `;
    }

    // Toggle Auto Streaming
    function toggleAutoStream() {
        isAutoStreaming = !isAutoStreaming;
        if (isAutoStreaming) {
            btnToggleStream.classList.remove('btn-secondary');
            btnToggleStream.classList.add('btn-primary');
            btnToggleStream.textContent = 'Auto Stream: ON (Active)';
            addAuditLog('[STREAM] Live graph event stream STARTED.', 'info');
            
            triggerSimulateSingle();
            autoStreamTimer = setInterval(() => {
                triggerSimulateSingle();
            }, 3000);
        } else {
            btnToggleStream.classList.remove('btn-primary');
            btnToggleStream.classList.add('btn-secondary');
            btnToggleStream.textContent = 'Auto Stream: OFF';
            if (autoStreamTimer) clearInterval(autoStreamTimer);
            addAuditLog('[STREAM] Live graph event stream PAUSED.', 'info');
        }
    }

    // Custom Transaction Form Submit
    evalForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const payload = {
            tx_id: document.getElementById('input-tx-id').value,
            btc_volume: parseFloat(document.getElementById('input-btc').value),
            inputs: parseInt(document.getElementById('input-inputs').value, 10),
            outputs: parseInt(document.getElementById('input-outputs').value, 10),
            probability: parseFloat(document.getElementById('input-prob').value),
            uncertainty: parseFloat(document.getElementById('input-unc').value)
        };

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                const tx = await res.json();
                addAuditLog(`[INFERENCE] Evaluated ${tx.tx_id} -> Action: ${tx.action}`, tx.probability >= 0.90 ? 'decline' : 'info');
                
                // Add to local history and render
                localHistory.unshift(tx);
                renderTxTable(localHistory);

                // Auto generate next random ID
                document.getElementById('input-tx-id').value = `txId_${Math.floor(Math.random() * 900000 + 100000)}`;
            }
        } catch (err) {
            addAuditLog(`[ERROR] Inference failed: ${err.message}`, 'error');
        }
    });

    // Modal: Open Modal
    function openReviewModal(tx) {
        activeReviewTxId = tx.tx_id;
        modalTxHash.textContent = tx.tx_id;
        
        const btcVal = tx.btc_volume || tx.btc || 0;
        const usdVal = Math.round(btcVal * 63500);
        modalAmount.textContent = `${btcVal.toFixed(2)} BTC ($${usdVal.toLocaleString()})`;
        
        modalRisk.textContent = tx.probability.toFixed(4);
        modalUnc.textContent = `±${tx.uncertainty.toFixed(4)}`;
        modalAction.textContent = tx.action || tx.status_code;

        modalDecisionSelect.value = (tx.tier === 1 || tx.probability >= 0.90) ? 'DECLINED' : 'APPROVED';
        modalNotes.value = '';
        reviewModal.classList.remove('hidden');
    }

    // Modal: Close Modal
    function closeReviewModal() {
        reviewModal.classList.add('hidden');
        activeReviewTxId = null;
    }

    // Modal: Submit Override
    async function submitOverride() {
        if (!activeReviewTxId) return;

        const newStatus = modalDecisionSelect.value;
        const reason = modalNotes.value.trim() || 'Compliance officer manual override';

        try {
            const res = await fetch('/api/override', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tx_id: activeReviewTxId,
                    new_status: newStatus,
                    reason: reason
                })
            });

            if (res.ok) {
                addAuditLog(`[OVERRIDE] ${activeReviewTxId} overridden to ${newStatus}. Reason: "${reason}"`, newStatus === 'APPROVED' ? 'approve' : 'decline');
                closeReviewModal();
                await fetchHistory();
            }
        } catch (err) {
            addAuditLog(`[ERROR] Override failed: ${err.message}`, 'error');
        }
    }

    // Event Listeners
    btnSimulateBatch.addEventListener('click', () => triggerSimulateBatch(3));
    btnToggleStream.addEventListener('click', toggleAutoStream);
    modalCloseBtn.addEventListener('click', closeReviewModal);
    modalCancelBtn.addEventListener('click', closeReviewModal);
    modalSubmitBtn.addEventListener('click', submitOverride);

    // Close modal on background click
    reviewModal.addEventListener('click', (e) => {
        if (e.target === reviewModal) closeReviewModal();
    });

    // Initial Initialization
    async function init() {
        addAuditLog('[INIT] Connected to Dir-Res GCN Backend on port 8050.', 'info');
        await fetchStats();
        await triggerSimulateBatch(4);
    }

    init();
});
