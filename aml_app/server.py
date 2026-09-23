import os
import sys
import json
import random
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import numpy as np

# Ensure directory is in path
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(app_dir)

from aml_illicit_transaction_blocker import AMLComplianceRouter

# Simulated Dataset Metrics & State
ROUTER = AMLComplianceRouter(prob_decline_threshold=0.932, prob_review_threshold=0.700, unc_threshold=0.040)

# Pre-seeded sample transaction pools
SAMPLE_TRANSACTIONS = [
    {"tx_id": "txId_904812", "btc": 142.50, "fee": 0.045, "inputs": 12, "outputs": 2, "prob": 0.9850, "unc": 0.0084, "gt": "Illicit (Darknet)"},
    {"tx_id": "txId_819234", "btc": 1.25, "fee": 0.0002, "inputs": 2, "outputs": 2, "prob": 0.1240, "unc": 0.0012, "gt": "Licit (Exchange)"},
    {"tx_id": "txId_774102", "btc": 45.80, "fee": 0.012, "inputs": 4, "outputs": 8, "prob": 0.8140, "unc": 0.0450, "gt": "Unknown (Peeling Chain)"},
    {"tx_id": "txId_651294", "btc": 0.45, "fee": 0.0001, "inputs": 1, "outputs": 2, "prob": 0.0850, "unc": 0.0009, "gt": "Licit (Merchant)"},
    {"tx_id": "txId_991823", "btc": 320.00, "fee": 0.089, "inputs": 25, "outputs": 1, "prob": 0.9920, "unc": 0.0041, "gt": "Illicit (Ransomware)"},
    {"tx_id": "txId_512903", "btc": 12.30, "fee": 0.0035, "inputs": 3, "outputs": 3, "prob": 0.7450, "unc": 0.0120, "gt": "Unknown (Mixer Deposit)"},
    {"tx_id": "txId_409182", "btc": 8.90, "fee": 0.0010, "inputs": 2, "outputs": 2, "prob": 0.2100, "unc": 0.0025, "gt": "Licit (User Transfer)"},
    {"tx_id": "txId_882910", "btc": 89.00, "fee": 0.025, "inputs": 8, "outputs": 15, "prob": 0.8950, "unc": 0.0520, "gt": "Illicit (Sanction Evader)"},
    {"tx_id": "txId_312094", "btc": 0.05, "fee": 0.00005, "inputs": 1, "outputs": 1, "prob": 0.0350, "unc": 0.0004, "gt": "Licit (Micro-payment)"},
    {"tx_id": "txId_604921", "btc": 56.40, "fee": 0.015, "inputs": 5, "outputs": 4, "prob": 0.7810, "unc": 0.0095, "gt": "Unknown (Tumbler Output)"}
]

TRANSACTION_LOG = []
COUNTER = 1000

class AMLAppRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=app_dir, **kwargs)

    def do_GET(self):
        if self.path == '/api/stats':
            self.send_json_response({
                "model": "Directional Residual Bayesian GNN (Dir-ResGCN)",
                "dataset": "Elliptic Bitcoin Transaction Network",
                "total_nodes": 203769,
                "total_edges": 234355,
                "continuous_coverage": "100.0%",
                "pr_auc": 0.8068,
                "auc_roc": 0.9315,
                "f1_score": 0.7673,
                "recall": 0.8205,
                "accuracy": 0.8697,
                "illicit_caught": 14504,
                "tier_breakdown": {
                    "declined": {"count": 14504, "pct": 21.49},
                    "manual_review": {"count": 1822, "pct": 2.70},
                    "approved": {"count": 51178, "pct": 75.81}
                }
            })
        elif self.path == '/api/simulate' or self.path == '/api/live':
            global COUNTER
            COUNTER += 1
            sample = random.choice(SAMPLE_TRANSACTIONS).copy()
            tx_id = f"txId_{random.randint(100000, 999999)}"
            
            # Add small random jitter to risk & uncertainty
            p = max(0.01, min(0.99, sample["prob"] + random.uniform(-0.05, 0.05)))
            u = max(0.0001, min(0.08, sample["unc"] + random.uniform(-0.005, 0.005)))
            btc = round(max(0.01, sample["btc"] * random.uniform(0.8, 1.3)), 4)
            
            decision = ROUTER.route_transaction(tx_id, p, u)
            tx_record = {
                "tx_id": tx_id,
                "timestamp": time.strftime("%H:%M:%S"),
                "btc_volume": btc,
                "fee": sample["fee"],
                "inputs": sample["inputs"],
                "outputs": sample["outputs"],
                "probability": round(p, 4),
                "uncertainty": round(u, 5),
                "action": decision["action"],
                "tier": decision["tier"],
                "status_code": decision["status_code"],
                "reason": decision["reason"],
                "gt_category": sample["gt"]
            }
            
            TRANSACTION_LOG.insert(0, tx_record)
            if len(TRANSACTION_LOG) > 100:
                TRANSACTION_LOG.pop()
                
            self.send_json_response(tx_record)
        elif self.path == '/api/history':
            self.send_json_response({"history": TRANSACTION_LOG})
        else:
            super().do_GET()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            body = json.loads(post_data.decode('utf-8'))
        except Exception:
            body = {}
            
        if self.path == '/api/predict':
            tx_id = body.get('tx_id', f"txId_{random.randint(100000, 999999)}")
            btc = float(body.get('btc_volume', 5.0))
            inputs = int(body.get('inputs', 2))
            outputs = int(body.get('outputs', 2))
            
            # Heuristic model mock based on inputs/outputs & volume
            base_p = min(0.98, (inputs * 0.05) + (btc * 0.002) + (0.5 if outputs > 8 else 0.1))
            p = float(body.get('probability', base_p))
            u = float(body.get('uncertainty', 0.005))
            
            decision = ROUTER.route_transaction(tx_id, p, u)
            res = {
                "tx_id": tx_id,
                "timestamp": time.strftime("%H:%M:%S"),
                "btc_volume": btc,
                "inputs": inputs,
                "outputs": outputs,
                "probability": round(p, 4),
                "uncertainty": round(u, 5),
                "action": decision["action"],
                "tier": decision["tier"],
                "status_code": decision["status_code"],
                "reason": decision["reason"]
            }
            self.send_json_response(res)
        elif self.path == '/api/override':
            tx_id = body.get('tx_id', '')
            new_status = body.get('new_status', 'APPROVED')
            reason = body.get('reason', 'Compliance officer manual override')
            
            # Update in log
            for item in TRANSACTION_LOG:
                if item['tx_id'] == tx_id:
                    item['status_code'] = new_status
                    item['action'] = f"MANUAL {new_status} BY OFFICER"
                    item['reason'] = reason
                    break
                    
            self.send_json_response({"success": True, "tx_id": tx_id, "new_status": new_status, "reason": reason})
        else:
            self.send_error(404, "Endpoint not found")

    def send_json_response(self, data):
        body = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

def run_server(port=8050):
    server_address = ('', port)
    httpd = HTTPServer(server_address, AMLAppRequestHandler)
    print(f"==========================================================================")
    print(f"   AGENTIC AML ILLICIT TRANSACTION DETECTION & DECLINE DASHBOARD SERVER   ")
    print(f"==========================================================================")
    print(f"   * Server Running at: http://localhost:{port}")
    print(f"   * Open http://localhost:{port} in your browser to view the application.")
    print(f"==========================================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == '__main__':
    port = 8050
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
