# IoT Network Security Report

**Generated:** November 30, 2025 at 05:07 PM

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Traffic Analyzed | 76,671 connections |
| Threat Level | **HIGH** |
| Malicious Traffic | 47,706 (62.2%) |
| Benign Traffic | 28,965 (37.8%) |

---

## Top Threats Detected

1. **PortScan**: 29,814 attacks (62.5%)
2. **C&C**: 3,331 attacks (7.0%)
3. **Attack**: 144 attacks (0.3%)
4. **DDoS**: 4,494 attacks (9.4%)
5. **FileDownload**: 10 attacks (0.0%)

---

## CVE Vulnerability Mapping

### CVE-2024-45163 (CVSS: 9.1)

The Mirai botnet through 2024-08-19 mishandles simultaneous TCP connections to the CNC (command and control) server. Unauthenticated sessions remain open, causing resource consumption. For example, an attacker can send a recognized username (such as root), or can send arbitrary data....

### CVE-1999-1247 (CVSS: nan)

Vulnerability in HP Camera component of HP DCE/9000 in HP-UX 9.x allows attackers to gain root privileges....

### CVE-2001-1543 (CVSS: nan)

Axis network camera 2120, 2110, 2100, 200+ and 200 contains a default administration password "pass", which allows remote attackers to gain access to the camera....

### CVE-2003-0240 (CVSS: nan)

The web-based administration capability for various Axis Network Camera products allows remote attackers to bypass access restrictions and modify configuration via an HTTP request to the admin/admin.shtml containing a leading // (double slash)....

### CVE-2004-1650 (CVSS: nan)

D-Link DCS-900 Internet Camera listens on UDP port 62976 for an IP address, which allows remote attackers to change the IP address of the camera via a UDP broadcast packet....

---

## Remediation Recommendations

1. 🔥 Immediately patch CVEs with CVSS scores above 7.0
2. 🛡️ Implement network segmentation to isolate IoT devices
3. 🚫 Block C&C server IPs at firewall level
4. 🔐 Change default credentials on all IoT devices
5. 📊 Enable continuous monitoring with automated alerts
6. ⚙️ Disable unnecessary services on IoT devices (telnet, SSH)
7. 🔄 Update firmware on all cameras and routers
8. 🌐 Restrict outbound connections from IoT devices

---

*This report was generated automatically by the IoT Malware Detection System.*
*Model Performance: Accuracy 99.56% | AUC-ROC 99.94%*
