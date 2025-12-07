"""
Format multi-stage analysis with live progress thinking
"""

def format_live_thinking(results):
    """Format response with live thinking progression"""
    
    response = f"""
🤔 **ANALYZING YOUR TRAFFIC...**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Analysis Timestamp:** {results['timestamp']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
    
    # Show each stage
    for stage in results['stages']:
        stage_num = stage['stage']
        stage_name = stage['name']
        
        response += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Step {stage_num}/3: {stage_name}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # Stage 1: Binary
        if stage_num == 1:
            models = ', '.join(stage['models_used'])
            response += f"""
🤖 Running **{models}** (binary classifier)...
   → Result: **{stage['prediction']}** ({stage['confidence']*100:.1f}% confidence)
   → Malicious probability: {stage['details']['malicious_prob']*100:.1f}%
   → Benign probability: {stage['details']['benign_prob']*100:.1f}%
"""
            if stage['prediction'] == 'Benign':
                response += "   ✅ Traffic is safe - no further analysis needed\n"
            else:
                response += f"   ⚠️ Threat detected - proceeding to attack classification\n"
        
        # Stage 2: Attack Type
        elif stage_num == 2:
            response += f"""
🧠 Running **Neural Network** (attack analyzer)...
   → Detected: **{stage['malware_family']}**
   → Confidence: {stage['confidence']*100:.1f}%
   ✅ Attack type identified - searching for vulnerabilities
"""
        
        # Stage 3: RAG
        elif stage_num == 3:
            num_cves = len(stage['cves'])
            critical = sum(1 for cve in stage['cves'] if cve['cvss_score'] >= 9.0)
            response += f"""
🔍 Searching **389 CVE database** for "{results['malware_family']}"...
   → Found **{num_cves}** related vulnerabilities
   → Critical threats (CVSS ≥ 9.0): **{critical}**
   ✅ Threat intelligence gathered
"""
    
    # Final Summary
    response += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ **ANALYSIS COMPLETE**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📊 **FINAL ASSESSMENT**

**Status:** {'🟢 BENIGN' if results['final_prediction'] == 'Benign' else '🔴 MALICIOUS'}
**Confidence:** {results['confidence']*100:.1f}%
"""
    
    if results['final_prediction'] == 'Malicious':
        response += f"**Threat Type:** {results['malware_family']}\n"
        
        # Show top CVEs
        if results['cves']:
            response += "\n## 🔓 **TOP VULNERABILITIES**\n\n"
            for i, cve in enumerate(results['cves'][:5], 1):
                response += f"{i}. **{cve['cve_id']}** (CVSS: {cve['cvss_score']})\n"
                response += f"   {cve['description']}...\n\n"
    
    # Recommendations
    response += "\n## 🛡️ **RECOMMENDED ACTIONS**\n\n"
    for i, rec in enumerate(results['recommendations'], 1):
        response += f"{i}. {rec}\n"
    
    response += "\n" + "━"*70
    response += "\n📄 **Would you like a detailed PDF report?**"
    
    return response
