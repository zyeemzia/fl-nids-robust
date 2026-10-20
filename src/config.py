import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR, PROC_DIR = ROOT / "data" / "raw", ROOT / "data" / "processed"
RESULTS_DIR, FIG_DIR = ROOT / "results", ROOT / "figures"
for p in (RAW_DIR, PROC_DIR, RESULTS_DIR, FIG_DIR):
    p.mkdir(parents=True, exist_ok=True)

SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,123,456").split(",")]
MAX_ROWS = int(os.environ.get("MAX_ROWS", "150000"))

N_CLIENTS = int(os.environ.get("N_CLIENTS", "20"))
ROUNDS = int(os.environ.get("ROUNDS", "30"))
LOCAL_EPOCHS = 2
LOCAL_LR = 0.01
BATCH = 64
PROX_MU = 0.01          # FedProx proximal weight
# Default off = the rule exactly as published. FEDNOVA_RENORM=1 enables an
# ablation that rescales the step; see agg_fednova() and docs/FEDNOVA.md.
FEDNOVA_RENORM = os.environ.get("FEDNOVA_RENORM", "0") == "1"

ALPHAS = [100.0, 1.0, 0.5, 0.1]        # Dirichlet: 100 ~ IID, 0.1 = severe skew
AGGREGATORS = ["fedavg", "fedprox", "scaffold", "fednova"]
MALICIOUS_FRACS = [0.0, 0.1, 0.2, 0.3]
ROBUST_AGGS = ["fedavg", "median", "trimmed_mean", "krum"]

# Edge-IIoTset identifier / capture-artifact columns (the authors drop these too).
DROP_COLS = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
             "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
             "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
             "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg"]
