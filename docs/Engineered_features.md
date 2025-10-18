# Final Engineered Features

This document describes the final feature set produced by the `build_dataset.py` script. This is the data that is fed directly into the machine learning models.

The total number of features is dynamic but is typically around **60-70**, depending on the number of unique categories found in `proto` and `conn_state`.

### 1. Base Numerical Features (9 features)
These are the original numerical columns, processed to handle missing values and high skew.
- **Processing Rule:**
    1. Invalid values are converted to `NaN`.
    2. `NaN` values are filled with `0`.
    3. Any negative values are clipped to `0`.
    4. A log transformation (`log(1 + x)`) is applied to skewed features.
- **Features:** `duration`, `orig_bytes`, `resp_bytes`, `orig_pkts`, `resp_pkts`, `orig_ip_bytes`, `resp_ip_bytes`, `missed_bytes`, `id.resp_p`.

### 2. Missing Value Flags (9 features)
A binary flag (`1` or `0`) is created for each base numerical feature to explicitly tell the model whether the original value was missing.
- **Example Features:** `duration_was_missing`, `orig_bytes_was_missing`, etc.

### 3. Label-Encoded Categorical Features (2 features)
High-cardinality categorical features are converted to integers.
- **Features:** `service`, `history`.

### 4. IP-Derived Features (10 features)
Features are extracted from both the originator (`id.orig_h`) and responder (`id.resp_h`) IP addresses.
- **Originator IP Features:** `orig_is_private`, `orig_is_broadcast`, etc.
- **Responder IP Features:** `resp_is_private`, `resp_is_broadcast`, etc.

### 5. Advanced Engineered Features (10 features)
These are new features created from combinations of the base features to capture more complex patterns.
- **Features:**
    - `is_port_23`: Binary flag for Telnet port.
    - `is_port_22`: Binary flag for SSH port.
    - `is_S0_state`: Binary flag for connections that only sent a SYN packet.
    - `is_telnet`: Placeholder feature (binary).
    - `is_unknown_service`: Placeholder feature (binary).
    - `upload_ratio`: Ratio of uploaded bytes to total bytes.
    - `bytes_per_packet`: Average bytes per packet for the connection.
    - `packet_rate`: Packets per second.
    - `is_scanning_signature`: Binary flag indicating a potential port scan (`port 23` and `S0` state).
    - `suspicious_score`: A simple heuristic score based on risky indicators.

### 6. One-Hot Encoded Categorical Features (~20-30 features)
Low-cardinality categorical features are converted into multiple binary columns.
- **Source Columns:** `proto`, `conn_state`.
- **Example Generated Features:** `proto_tcp`, `proto_udp`, `conn_state_SF`, `conn_state_S0`, etc.