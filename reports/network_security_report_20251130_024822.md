# IoT Network Security Report

**Generated:** November 30, 2025 at 02:48 AM

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
2. **Okiru**: 9,882 attacks (20.7%)
3. **DDoS**: 4,494 attacks (9.4%)

---

## CVE Vulnerability Mapping

### CVE-2016-10401 (CVSS: 9.8)

Mirai botnet telnet exploit allowing remote code execution through default credentials...

### CVE-2017-17215 (CVSS: 8.8)

Router remote code execution vulnerability exploited by IoT botnets...

---

## Remediation Recommendations

1. Patch all CVEs with CVSS scores above 7.0 immediately
2. Block identified C&C server IP addresses at firewall level
3. Change default passwords on all IoT devices

---

*This report was generated automatically by the IoT Malware Detection System.*
*Model Performance: Accuracy 99.56% | AUC-ROC 99.94%*
