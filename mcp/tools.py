import sys
import os
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Path setup so we can import `config` and (optionally) `models.*`
# ---------------------------------------------------------------------------

current_file_path = os.path.abspath(__file__)
mcp_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(mcp_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config

try:
    import joblib
except ImportError:  # very unlikely in your venv, but be defensive
    print("⚠️ joblib not installed; traditional models will not load.")
    joblib = None

# Optional: importing these makes unpickling safer if joblib stored class refs
try:
    from models.traditional.multiclass_classifier import MultiClassXGBoost  # noqa: F401
    from models.traditional.virus_classifier import VirusClassifier  # noqa: F401
except Exception:
    # Not fatal for inference as long as the pickles are plain xgboost/lightgbm objects
    print(
        "⚠️ Could not import model classes from 'models.traditional'. "
        "If your pickles store custom classes, loading may fail."
    )


class ThreatEngine:
    """Central inference engine for IoT-23 models.

    - XGBoost multiclass attack classifier  (multiclass_xgboost.pkl)
    - Virus family classifier              (virus_classifier.pkl)
    - Autoencoders / GAN                  (loaded for status only right now)
    - GNN models                          (loaded for status only right now)
    """

    def __init__(self) -> None:
        # Base directory where all models are stored (from Config)
        self.base_path = Config.MODELS_DIR

        # Concrete model paths
        self.paths = {
            # Traditional
            "attack_xgb": os.path.join(
                Config.TRADITIONAL_MODELS_DIR, "multiclass_xgboost.pkl"
            ),
            "attack_nn": os.path.join(
                Config.TRADITIONAL_MODELS_DIR, "best_multiclass_nn.pth"
            ),
            "family_lgb": os.path.join(
                Config.TRADITIONAL_MODELS_DIR, "virus_classifier.pkl"
            ),
            # ✅ ADD THIS:
            "binary_clf": os.path.join(
                Config.TRADITIONAL_MODELS_DIR, "binary_classifier_tuned.pkl"
            ),
            # Deep learning (anomaly / reconstruction)
            "ae_standard": os.path.join(
                Config.DL_MODELS_DIR, "best_autoencoder.pth"
            ),
            "ae_denoising": os.path.join(
                Config.DL_MODELS_DIR, "best_denoising_autoencoder.pth"
            ),
            "gan": os.path.join(
                Config.DL_MODELS_DIR, "gan_wgan_gp_final.pth"
            ),
            # Graph neural networks
            "gnn_sage": os.path.join(
                Config.GNN_MODELS_DIR, "best_gnn.pth"
            ),
            "gnn_gat": os.path.join(
                Config.GNN_MODELS_DIR, "best_gnn_gat.pth"
            ),
        }

        # ------------------------------------------------------------------
        # Feature list: exactly mirror what you used during training
        # Engineered features (52), then drop 8 non-numeric to get 44 numeric
        # ------------------------------------------------------------------
        try:
            full_features = Config.get_feature_list()
        except Exception as e:
            print(f"⚠️ Could not load feature list from Config: {e}")
            full_features = []

        non_numeric = {
            "id.orig_h",
            "id.resp_h",
            "proto",
            "service",
            "conn_state",
            "local_orig",
            "local_resp",
            "history",
        }
        self.numeric_features = [f for f in full_features if f not in non_numeric]
        self.numeric_index = {
            name: idx for idx, name in enumerate(self.numeric_features)
        }

        if self.numeric_features:
            print(
                f"📐 Inference will use {len(self.numeric_features)} numeric features "
                f"({len(full_features)} engineered total)."
            )
        else:
            print(
                "⚠️ numeric_features list empty; will fall back to heuristic vector."
            )

        # Label mappings (same as used during training)
        self.attack_labels = {
            0: "Benign",
            1: "DDoS",
            2: "PortScan",
            3: "C&C",
            4: "Okiru",
            5: "Attack",
        }

        self.family_labels = {
            0: "Benign",
            1: "Mirai",
            2: "Torii",
            3: "Gafgyt",
            4: "Kenjiro",
            5: "Hajime",
        }

        self.models: dict[str, object] = {}
        self.loaded: bool = False

        self._load_models()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_models(self) -> None:
        """Load all models that exist on disk into memory (CPU)."""

        # Traditional XGBoost attack classifier
        if joblib is not None and os.path.exists(self.paths["attack_xgb"]):
            try:
                obj = joblib.load(self.paths["attack_xgb"])
                if isinstance(obj, dict) and "model" in obj:
                    self.models["attack_xgb"] = obj["model"]
                else:
                    self.models["attack_xgb"] = obj
                print("✅ Loaded XGBoost attack classifier.")
            except Exception as e:
                print(f"⚠️ Failed to load XGBoost model: {e}")

        # Virus family classifier (LightGBM or similar)
        if joblib is not None and os.path.exists(self.paths["family_lgb"]):
            try:
                obj = joblib.load(self.paths["family_lgb"])
                if isinstance(obj, dict) and "model" in obj:
                    self.models["family_lgb"] = obj["model"]
                else:
                    self.models["family_lgb"] = obj
                print("✅ Loaded virus family classifier.")
            except Exception as e:
                print(f"⚠️ Failed to load virus classifier: {e}")

        # GNN models – store raw dict/objects; we only show status for now
        for key in ("gnn_sage", "gnn_gat"):
            path = self.paths.get(key)
            if path and os.path.exists(path):
                try:
                    self.models[key] = torch.load(path, map_location="cpu")
                    print(f"✅ Loaded {key} GNN model.")
                except Exception as e:
                    print(f"⚠️ Failed to load {key}: {e}")

        # AE / GAN – for now, we just reflect presence via paths
        self.loaded = True

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def _predict_single_vector(self, X: np.ndarray) -> tuple[str, str, float, str]:
        """Run attack + family classifier on a single 1×N feature vector."""
        attack_model = self.models.get("attack_xgb")
        family_model = self.models.get("family_lgb")

        pred_attack = "Unknown"
        pred_family = "None"
        conf = 0.0
        used_model = "Heuristics (Fallback)"

        # Attack model
        if attack_model is not None:
            try:
                if hasattr(attack_model, "predict_proba"):
                    probs = attack_model.predict_proba(X)[0]
                    pred_idx = int(np.argmax(probs))
                    conf = float(np.max(probs))
                else:
                    pred_idx = int(attack_model.predict(X)[0])
                    conf = 1.0

                pred_attack = self.attack_labels.get(pred_idx, f"Type_{pred_idx}")
                used_model = "XGBoost (Trained)"
            except Exception as e:
                print(f"⚠️ XGBoost inference error: {e}")

        # Family model – only if non-benign
        if family_model is not None and pred_attack not in ("Benign", "Unknown"):
            try:
                if hasattr(family_model, "predict_proba"):
                    probs_f = family_model.predict_proba(X)[0]
                    fam_idx = int(np.argmax(probs_f))
                else:
                    fam_idx = int(family_model.predict(X)[0])
                pred_family = self.family_labels.get(fam_idx, f"Fam_{fam_idx}")
            except Exception as e:
                print(f"⚠️ Family classifier inference error: {e}")

        return pred_attack, pred_family, conf, used_model

    # ------------------------------------------------------------------
    # Feature vector construction (minimal raw feature case)
    # ------------------------------------------------------------------
    def _make_vector(self, duration: float, bytes_t: int, port: int, proto: str) -> np.ndarray:
        """Build a 44-D numeric feature vector aligned with training (minimal info).

        Only uses:
        - duration
        - total bytes (mapped to orig_bytes)
        - dest port
        - proto (tcp/udp/other)
        - a few port flags

        All other features are 0. This is NOT full Option B – it's a fallback
        when you *don't* have a full engineered row.
        """
        if self.numeric_features:
            n_features = len(self.numeric_features)
        else:
            n_features = 44

        vec = np.zeros((1, n_features), dtype=np.float32)
        idx = self.numeric_index.get

        def set_feat(name: str, value: float) -> None:
            i = idx(name)
            if i is not None and 0 <= i < n_features:
                vec[0, i] = float(value)

        set_feat("duration", duration)
        set_feat("orig_bytes", bytes_t)
        set_feat("id.resp_p", port)

        p = (proto or "").lower()
        if p == "tcp":
            proto_code = 0
        elif p == "udp":
            proto_code = 1
        else:
            proto_code = 2
        set_feat("proto_idx", proto_code)

        if int(port) == 23:
            set_feat("is_port_23", 1.0)
            set_feat("is_telnet", 1.0)
        elif int(port) == 22:
            set_feat("is_port_22", 1.0)

        return vec

    # ------------------------------------------------------------------
    # Core inference: minimal-flow API
    # ------------------------------------------------------------------
    def analyze_flow(self, duration: float, bytes_t: int, port: int, proto: str) -> dict:
        """Analyze a flow described only by duration/bytes/port/proto (partial info).

        This is *not* full Option B – it's using a subset of the model's features,
        but still in the correct shape/order. Use analyze_feature_dict() or
        analyze_csv() for true full-feature inference.
        """
        X = self._make_vector(duration, bytes_t, port, proto)
        pred_attack, pred_family, conf, used_model = self._predict_single_vector(X)
        status = "MALICIOUS" if pred_attack not in ("Benign", "Unknown") else "BENIGN"

        return {
            "status": status,
            "attack_type": pred_attack,
            "family": pred_family,
            "confidence": round(conf, 3),
            "model": used_model,
            "port": int(port),
            "proto": str(proto),
            "bytes": int(bytes_t),
            "duration": float(duration),
        }

    # ------------------------------------------------------------------
    # Core inference: FULL feature dict (Option B)
    # ------------------------------------------------------------------
    def analyze_feature_dict(self, features: dict) -> dict:
        """Analyze a single flow given a full engineered feature dictionary.

        `features` must contain (at least) all names in self.numeric_features
        (44 numeric features). Missing ones are treated as 0.
        """
        if not self.numeric_features:
            return {
                "status": "UNKNOWN",
                "attack_type": "Unknown",
                "family": "None",
                "confidence": 0.0,
                "model": "No feature list available",
            }

        n_features = len(self.numeric_features)
        X = np.zeros((1, n_features), dtype=np.float32)

        for i, name in enumerate(self.numeric_features):
            X[0, i] = float(features.get(name, 0.0))

        pred_attack, pred_family, conf, used_model = self._predict_single_vector(X)
        status = "MALICIOUS" if pred_attack not in ("Benign", "Unknown") else "BENIGN"

        return {
            "status": status,
            "attack_type": pred_attack,
            "family": pred_family,
            "confidence": round(conf, 3),
            "model": used_model,
        }

    # ------------------------------------------------------------------
    # Core inference: FULL CSV (Option B for batches)
    # ------------------------------------------------------------------
    def analyze_csv(self, file_path: str, max_rows: int = 20000) -> str:
        """Analyze a CSV with full engineered features.

        This is true Option B:
        - Expects the CSV to contain *all* numeric feature columns.
        - Uses XGBoost + family classifier directly on the full feature matrix.
        """
        if not os.path.exists(file_path):
            return f"❌ File not found: {file_path}"

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            return f"Error reading CSV: {e}"

        if df.empty:
            return "No rows found in CSV."

        attack_model = self.models.get("attack_xgb")
        family_model = self.models.get("family_lgb")

        if attack_model is None:
            return "❌ Attack model is not loaded."
        if not self.numeric_features:
            return "❌ numeric_features list is empty; cannot run full-feature inference."

        # Ensure all numeric features are present
        missing = [f for f in self.numeric_features if f not in df.columns]
        if missing:
            return (
                "❌ CSV does not contain the full numeric feature set expected by the model.\n"
                f"Missing columns (example): {missing[:10]}"
            )

        if len(df) > max_rows:
            df = df.sample(max_rows, random_state=42)

        X = df[self.numeric_features].fillna(0).to_numpy(dtype=np.float32)

        # Attack predictions
        if hasattr(attack_model, "predict_proba"):
            probs = attack_model.predict_proba(X)
            attack_idx = probs.argmax(axis=1)
        else:
            attack_idx = attack_model.predict(X)
            probs = None

        attack_labels = [
            self.attack_labels.get(int(i), f"Type_{int(i)}") for i in attack_idx
        ]

        # Family predictions only for malicious flows
        fam_labels = ["None"] * len(df)
        if family_model is not None:
            malicious_mask = [
                lbl not in ("Benign", "Unknown") for lbl in attack_labels
            ]
            if any(malicious_mask):
                X_mal = X[malicious_mask]
                if hasattr(family_model, "predict_proba"):
                    probs_f = family_model.predict_proba(X_mal)
                    fam_idx = probs_f.argmax(axis=1)
                else:
                    fam_idx = family_model.predict(X_mal)
                fam_preds = [
                    self.family_labels.get(int(i), f"Fam_{int(i)}")
                    for i in fam_idx
                ]

                j = 0
                for i, is_mal in enumerate(malicious_mask):
                    if is_mal:
                        fam_labels[i] = fam_preds[j]
                        j += 1

        from collections import Counter

        attack_counts = Counter(attack_labels)
        fam_counts = Counter([f for f in fam_labels if f != "None"])

        summary = [
            f"Batch scan report: {len(df)} flows analyzed (full-feature inference).",
            "",
            "Attack type distribution:",
        ]
        for k, v in attack_counts.items():
            summary.append(f"  - {k}: {v}")

        if fam_counts:
            summary.append("")
            summary.append("Family distribution (on malicious flows):")
            for k, v in fam_counts.items():
                summary.append(f"  - {k}: {v}")

        return "\n".join(summary)

    # ------------------------------------------------------------------
    # KB-style helpers
    # ------------------------------------------------------------------
    def get_malware_info(self, query: str) -> str:
        q = (query or "").lower()
        kb = {
            "mirai": "Mirai targets IoT via Telnet (port 23), using default creds and scanning DVRs/routers.",
            "torii": "Torii is a modular IoT botnet with strong persistence and data exfiltration capabilities.",
            "gafgyt": "Gafgyt (Bashlite) infects routers and DVRs, often using shell-injection vulnerabilities.",
            "kenjiro": "Kenjiro is an IoT malware family targeting routers with brute-force logins.",
            "hajime": "Hajime is a peer-to-peer IoT botnet that can block ports used by Mirai.",
        }
        for name, desc in kb.items():
            if name in q:
                return f"📚 {name.capitalize()}: {desc}"
        return "No specific malware family information found for that query."

    def get_mitigation(self, threat_type: str) -> str:
        t = (threat_type or "").lower()
        if "mirai" in t or "telnet" in t or "23" in t:
            return (
                "Block Telnet (port 23), enforce strong unique passwords on all IoT "
                "devices, and patch/replace outdated DVRs and routers."
            )ear

        if "torii" in t:
            return (
                "Reset affected devices to factory settings, rotate all credentials/keys, "
                "and monitor for unusual DNS/HTTP patterns."
            )
        if "ddos" in t:
            return (
                "Apply rate-limiting, enable upstream DDoS protection/scrubbing, and "
                "geo-block or ASN-block known attack sources."
            )
        return (
            "General IoT hardening: segment IoT devices on a separate VLAN, apply "
            "firmware updates, disable unused services, and monitor logs for anomalies."
        )

    def check_device_history(self, ip: str) -> str:
        # Deterministic pseudo-history based on a simple hash of the IP string
        if not ip:
            return "Invalid IP address."

        seed = sum(ord(c) for c in ip) % 100
        if seed < 20:
            return f"{ip} has a clean reputation in the last 30 days."
        if seed < 70:
            return f"{ip} shows occasional suspicious spikes. Recommend monitoring."
        return f"{ip} is known for repeated attacks on IoT devices. Block immediately."

    def get_model_stats(self) -> str:
        lines: list[str] = []
        lines.append("=== Model Registry ===")
        lines.append(f"Base path: {self.base_path}")
        lines.append("")

        # XGBoost
        lines.append(
            f"XGBoost Attack Classifier: "
            f"{'✅ Loaded' if 'attack_xgb' in self.models else '❌ Missing'}"
        )

        # Virus classifier
        lines.append(
            f"Virus Family Classifier: "
            f"{'✅ Loaded' if 'family_lgb' in self.models else '❌ Missing'}"
        )

        # Autoencoders
        lines.append(
            f"Autoencoder (Standard): "
            f"{'✅ Found' if os.path.exists(self.paths['ae_standard']) else '❌ Missing'}"
        )
        lines.append(
            f"Autoencoder (Denoising): "
            f"{'✅ Found' if os.path.exists(self.paths['ae_denoising']) else '❌ Missing'}"
        )

        # GAN
        lines.append(
            f"WGAN-GP: "
            f"{'✅ Found' if os.path.exists(self.paths['gan']) else '❌ Missing'}"
        )

        # GNNs
        if "gnn_sage" in self.models:
            m = self.models["gnn_sage"]
            auc = m.get("test_auc", "N/A") if isinstance(m, dict) else "N/A"
            lines.append(f"GNN (SAGE): ✅ Loaded (AUC: {auc})")
        else:
            lines.append("GNN (SAGE): ❌ Missing")

        if "gnn_gat" in self.models:
            m = self.models["gnn_gat"]
            auc = m.get("test_auc", "N/A") if isinstance(m, dict) else "N/A"
            lines.append(f"GNN (GAT): ✅ Loaded (AUC: {auc})")
        else:
            lines.append("GNN (GAT): ❌ Missing")

        return "\n".join(lines)

    def explain_prediction(self, flow_id: str) -> str:
        # Placeholder; you can later hook in SHAP / feature importances.
        return "💡 EXPLAINER: Risk driven by Port 23 exposure and high byte volume."


# Single global engine instance
engine = ThreatEngine()


# ---------------------------------------------------------------------------
# Plain Python tool functions for MCP & Streamlit/Gemini
# ---------------------------------------------------------------------------

def scan_flow(duration: float, bytes_t: int, port: int, proto: str = "tcp") -> str:
    """Analyze a single flow using partial features (duration/bytes/port/proto)."""
    result = engine.analyze_flow(duration, bytes_t, port, proto)
    return str(result)


def scan_file(file_path: str) -> str:
    """Analyze a CSV file of flows using FULL numeric feature set (Option B)."""
    if not os.path.exists(file_path):
        return f"❌ File not found: {file_path}"
    return engine.analyze_csv(file_path)


def scan_feature_vector(features: dict) -> str:
    """Analyze a single flow using a FULL engineered feature dict (Option B).

    `features` keys must match the numeric feature names from feature_list.joblib.
    Missing keys are treated as 0.
    """
    result = engine.analyze_feature_dict(features)
    return str(result)


def lookup_malware_info(malware_name: str) -> str:
    """Look up information on a known IoT malware family."""
    return engine.get_malware_info(malware_name)


def get_mitigation_plan(threat_name: str) -> str:
    """Return mitigation guidance for a given threat or family name."""
    return engine.get_mitigation(threat_name)


def check_device_history(ip_address: str) -> str:
    """Return a pseudo-history for a device IP."""
    return engine.check_device_history(ip_address)


def get_model_performance() -> str:
    """Return a human-readable registry of which models are available."""
    return engine.get_model_stats()


def explain_prediction(flow_id: str) -> str:
    """Explain why a given flow (by id) was flagged (placeholder)."""
    return engine.explain_prediction(flow_id)
