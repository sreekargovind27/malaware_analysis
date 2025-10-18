# FILE_STRUCTURE.md

# IoT-23 Dataset File Structure

## Expected Data Files (21 total)

### Benign Files (2)
```
data/raw/CTU-Honeypot-Capture-4-1.csv
data/raw/CTU-Honeypot-Capture-5-1.csv
```

### Malicious Files (19)
```
data/raw/CTU-IoT-Malware-Capture-1-1.csv
data/raw/CTU-IoT-Malware-Capture-7-1.csv
data/raw/CTU-IoT-Malware-Capture-8-1.csv
data/raw/CTU-IoT-Malware-Capture-9-1.csv
data/raw/CTU-IoT-Malware-Capture-17-1.csv
data/raw/CTU-IoT-Malware-Capture-18-1.csv
data/raw/CTU-IoT-Malware-Capture-19-1.csv
data/raw/CTU-IoT-Malware-Capture-20-1.csv
data/raw/CTU-IoT-Malware-Capture-21-1.csv
data/raw/CTU-IoT-Malware-Capture-33-1.csv
data/raw/CTU-IoT-Malware-Capture-34-1.csv
data/raw/CTU-IoT-Malware-Capture-35-1.csv
data/raw/CTU-IoT-Malware-Capture-36-1.csv
data/raw/CTU-IoT-Malware-Capture-39-1.csv
data/raw/CTU-IoT-Malware-Capture-42-1.csv
data/raw/CTU-IoT-Malware-Capture-43-1.csv
data/raw/CTU-IoT-Malware-Capture-44-1.csv
data/raw/CTU-IoT-Malware-Capture-48-1.csv
data/raw/CTU-IoT-Malware-Capture-49-1.csv
data/raw/CTU-IoT-Malware-Capture-52-1.csv
data/raw/CTU-IoT-Malware-Capture-60-1.csv
```

## Column Structure (23 columns per file)

```
Source_Folder
ts
uid
id.orig_h
id.orig_p
id.resp_h
id.resp_p
proto
service
duration
orig_bytes
resp_bytes
conn_state
local_orig
local_resp
missed_bytes
history
orig_pkts
orig_ip_bytes
resp_pkts
resp_ip_bytes
label
detailed-label
```