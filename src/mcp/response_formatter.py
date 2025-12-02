"""
Response Formatter - Creates user-friendly output from analysis results
Formats comprehensive analysis into readable markdown text
"""

from typing import Dict, List


def format_comprehensive_response(results: Dict) -> str:
    """
    Format comprehensive analysis into readable markdown report

    Args:
        results: Complete analysis results from orchestrator

    Returns:
        Formatted markdown string
    """

    output = []

    # Header
    output.append("=" * 70)
    output.append("🔍 COMPREHENSIVE IOT TRAFFIC SECURITY ANALYSIS")
    output.append("=" * 70)
    output.append(f"**Analysis Timestamp:** {results['timestamp']}")
    output.append("")

    # Classification Section
    output.append("## 📊 CLASSIFICATION RESULTS")
    output.append("-" * 70)

    classification = results['classification']
    pred = classification['prediction']
    conf = classification['confidence']

    if pred == "Malicious":
        output.append(f"**Status:** 🚨 **MALICIOUS TRAFFIC DETECTED**")
        output.append(f"**Confidence:** {conf:.1%}")
        output.append(f"**Threat Type:** {classification['malware_family']}")
    else:
        output.append(f"**Status:** ✅ **BENIGN TRAFFIC**")
        output.append(f"**Confidence:** {conf:.1%}")

    output.append(f"**Malicious Probability:** {classification['malicious_probability']:.1%}")
    output.append(f"**Benign Probability:** {classification['benign_probability']:.1%}")

    traffic = classification['traffic_details']
    output.append(f"\n**Traffic Details:**")
    output.append(f"- Ports: {traffic['orig_port']} → {traffic['resp_port']}")
    output.append(f"- Protocol: {traffic['protocol'].upper()}")
    output.append("")

    # If malicious, show detailed threat intelligence
    if pred == "Malicious":

        # Risk Assessment
        output.append("## 🎯 RISK ASSESSMENT")
        output.append("-" * 70)

        risk = results['risk_assessment']
        output.append(f"**Threat Level:** {risk['color']} **{risk['level']}**")
        output.append(f"**Maximum CVSS Score:** {risk['max_cvss']:.1f}/10.0")
        output.append(f"**Average CVSS Score:** {risk['avg_cvss']:.1f}/10.0")
        output.append(f"**Exploitability:** {risk['exploitability']}")
        output.append(f"**Impact:** {risk['impact']}")
        output.append(f"**Urgency:** {risk['urgency']}")
        output.append("")

        # Vulnerabilities Section
        output.append("## 🔓 RELATED VULNERABILITIES (CVE DATABASE)")
        output.append("-" * 70)

        vulns = results['vulnerabilities']
        output.append(f"**Total CVEs Found:** {vulns['total_found']}")
        output.append(f"**Critical (CVSS ≥ 9.0):** {len(vulns['critical'])}")
        output.append(f"**High (CVSS 7.0-8.9):** {len(vulns['high'])}")
        output.append(f"**Medium (CVSS 4.0-6.9):** {len(vulns['medium'])}")
        output.append("")

        # Top 5 CVEs
        output.append("**🔥 Top 5 Critical Vulnerabilities:**")
        output.append("")
        for i, cve in enumerate(vulns['top_5'], 1):
            severity = "🔴 CRITICAL" if cve['cvss_score'] >= 9.0 else "🟠 HIGH" if cve[
                                                                                     'cvss_score'] >= 7.0 else "🟡 MEDIUM"
            output.append(f"**{i}. {cve['cve_id']}** ({severity} - CVSS {cve['cvss_score']:.1f})")
            output.append(f"   {cve['description'][:200]}...")
            output.append("")

        # Attack Patterns
        output.append("## 🎭 ATTACK PATTERN ANALYSIS")
        output.append("-" * 70)

        patterns = results['attack_patterns']
        output.append(f"**Malware Family:** {patterns['malware_family']}")
        output.append(f"**Samples Analyzed:** {patterns['sample_count']:,}")
        output.append(f"**Primary Protocol:** {patterns['primary_protocol'].upper()}")
        output.append("")

        output.append("**Protocol Distribution:**")
        for proto, count in patterns['protocols'].items():
            output.append(f"- {proto.upper()}: {count:,} connections")
        output.append("")

        output.append("**Common Target Ports:**")
        for port_info in patterns['common_target_ports'][:5]:
            output.append(f"- Port {port_info['port']}: {port_info['count']:,} attempts")
        output.append("")

        # Prevalence
        output.append("## 📈 THREAT PREVALENCE")
        output.append("-" * 70)

        prev = results['prevalence']
        output.append(f"**This Threat:** {prev['family_count']:,} detections")
        output.append(f"**Total Threats:** {prev['total_malicious']:,} malicious samples")
        output.append(f"**Percentage:** {prev['percentage_of_threats']:.1f}% of all threats")
        output.append(f"**Threat Ranking:** {prev['rank_suffix']} most common threat")
        output.append("")

        # Recommendations
        output.append("## 🛡️ REMEDIATION RECOMMENDATIONS")
        output.append("-" * 70)
        output.append("**IMMEDIATE ACTIONS REQUIRED:**")
        output.append("")

        for i, rec in enumerate(results['recommendations'], 1):
            output.append(f"{i}. {rec}")

        output.append("")

    else:  # Benign traffic
        output.append("## ✅ SECURITY STATUS")
        output.append("-" * 70)
        output.append("**Assessment:** Traffic appears legitimate and safe")
        output.append("**Risk Level:** LOW")
        output.append("")
        output.append("**Recommendations:**")
        for i, rec in enumerate(results['recommendations'], 1):
            output.append(f"{i}. {rec}")
        output.append("")

    # Footer
    output.append("=" * 70)
    output.append("📄 **DETAILED REPORT AVAILABLE**")
    output.append("")
    output.append("Would you like me to generate a comprehensive PDF report?")
    output.append("(Includes executive summary, charts, full CVE details, and remediation plan)")
    output.append("=" * 70)
    output.append("")
    output.append("*Analysis powered by ML model (99.56% accuracy) + RAG-enhanced threat intelligence*")

    return "\n".join(output)


def format_quick_summary(results: Dict) -> str:
    """
    Generate concise summary for immediate display

    Args:
        results: Analysis results

    Returns:
        Short summary string
    """

    classification = results['classification']
    pred = classification['prediction']

    if pred == "Malicious":
        risk = results['risk_assessment']
        summary = f"""
🚨 **THREAT DETECTED**

Type: {classification['malware_family']}
Confidence: {classification['confidence']:.1%}
Risk Level: {risk['color']} {risk['level']}

Top Vulnerability: {results['vulnerabilities']['top_5'][0]['cve_id']} (CVSS {results['vulnerabilities']['top_5'][0]['cvss_score']:.1f})

Immediate action required - see full analysis below.
"""
    else:
        summary = f"""
✅ **TRAFFIC IS SAFE**

Confidence: {classification['confidence']:.1%}
Risk Level: LOW

No threats detected. Continue normal monitoring.
"""

    return summary