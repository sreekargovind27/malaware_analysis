# MCP Tools Specification

## Tool 1: classify_traffic_malicious
**Type:** Spark Model Tool (LightGBM)
**Input:** 
- connection_log: dict with fields (id.orig_h, id.resp_h, proto, orig_bytes, etc.)
**Output:** 
- prediction: "Benign" or "Malicious"
- confidence: float (0.0 to 1.0)
**Model:** LightGBM trained on full IoT-23 dataset

## Tool 2: cluster_malware_behavior  
**Type:** Spark Model Tool (K-Means)
**Input:**
- connection_log: dict with malicious traffic features
**Output:**
- cluster_id: int (0 to k-1)
- cluster_description: string explaining behavior type
**Model:** K-Means trained on malicious IoT-23 traffic

## Tool 3: get_anomaly_score
**Type:** Data Analytics Tool (Autoencoder)
**Input:**
- connection_log: dict with traffic features
**Output:**
- anomaly_score: float (reconstruction error)
- is_anomaly: boolean (True if score > threshold)
**Model:** Autoencoder trained on benign traffic

## Tool 4: analyze_attack_patterns
**Type:** Data Analytics Tool (Spark SQL)
**Input:**
- malware_family: string (e.g., "Mirai", "Okiru")
**Output:**
- statistics: dict with attack characteristics
  - common_ports: list
  - avg_packets: float
  - geographic_origins: list
**Data:** Aggregated from IoT-23 dataset

## Tool 5: get_device_threat_profile
**Type:** Integrated Intelligence Tool (GNN)
**Input:**
- ip_address: string
**Output:**
- predicted_malware_family: string or "Benign"
- confidence: float
- behavioral_note: string
**Model:** GNN trained on IoT-23 graph structure

## Tool 6: check_vulnerability_mapping
**Type:** Integrated Intelligence Tool (CVE Database)
**Input:**
- malware_family: string OR device_type: string
**Output:**
- related_cves: list of CVE IDs
- vulnerability_descriptions: list
- remediation_steps: list
**Data:** CVE database joined with malware family mappings