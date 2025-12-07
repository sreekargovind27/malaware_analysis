# MCP Server - IoT Threat Detection

**Author:** Nidhi Rajani, Arun Ghontale, Govind Sreekar Garimella

**Phase 3 Component:** Model Context Protocol Integration  
**Course:** EAS 587 Fall 2025

## Overview

This directory contains the complete MCP (Model Context Protocol) server implementation that makes our ML models accessible through natural language via Claude Desktop.

## What I Built

### Core MCP Server (`src/mcp/`)
- **mcp_server_final.py** - Main MCP server with 11 security tools
- **multi_stage_orchestrator.py** - Orchestrates 3-stage ML pipeline
- **model_loaders.py** - Loads Spark MLlib, PyTorch, and ChromaDB models
- **input_validator.py** - Validates network traffic input
- **response_formatter.py** - Formats analysis results for Claude
- **analysis_orchestrator.py** - Coordinates all tool calls
- **report_generator.py** - Generates PDF reports (Tool 10)

### 11 MCP Tools Implemented

1. **validate_traffic_data** - Input validation & sanitization
2. **classify_traffic_malicious** - Binary detection (Stage 1: LightGBM + VAE + GNN)
3. **predict_attack_type** - Attack categorization (Stage 2: Neural Net)
4. **identify_malware_family** - Malware attribution (Stage 3: LightGBM)
5. **search_vulnerability_knowledge** - RAG search through 389 CVE lakehouse
6. **analyze_attack_patterns** - Dataset statistics & prevalence
7. **get_threat_recommendations** - CVE-backed remediation steps
8. **calculate_risk_score** - CRITICAL/HIGH/MEDIUM/LOW assessment
9. **get_attack_statistics** - Historical context from training data
10. **generate_report** - PDF report generation
11. **analyze_traffic_multistage** - Complete orchestration (calls tools 1-9)

### RAG System (`utils/data/`)
- **build_cve_rag.py** - Builds ChromaDB vector database from 389 CVEs
- **download_cve_data.py** - Downloads CVEs from NIST database

### Testing (`test/`)
- **test_comprehensive_analysis.py** - Full pipeline test with all scenarios
- **test_mcp_manual.py** - Manual testing script for individual tools
- **test_real_samples.py** - Tests with real malware samples

## Quick Start

### Installation
```bash
# Install MCP-specific dependencies
pip install -r requirements_mcp.txt
```

### Running MCP Server
```bash
cd src/mcp
python mcp_server_final.py
```

### Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "iot-threat-detection": {
      "command": "python",
      "args": ["/absolute/path/to/malaware_analysis/src/mcp/mcp_server_final.py"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/malaware_analysis"
      }
    }
  }
}
```

Restart Claude Desktop.

### Testing
```bash
# Test all scenarios
python test/test_comprehensive_analysis.py

# Test specific malware
python test/test_real_samples.py --type mirai

# Manual tool testing
python test/test_mcp_manual.py
```

## Usage Examples

### Example 1: Mirai PortScan Detection

**Input in Claude Desktop:**
```
Analyze this traffic: port 23, duration 0.002s, 40 bytes, TCP
```

**Output:**
```
Classification: MALICIOUS (98.9% confidence)
Attack Type: PortScan
Malware: Mirai
Risk: CRITICAL
CVEs: CVE-2001-0554 (CVSS 9.8), CVE-2016-10401 (CVSS 9.3)
Recommendations: [8 specific actions with CVE references]
Time: 18 seconds
```

### Example 2: Benign NTP Traffic

**Input:**
```
Analyze: port 123, duration 5.5s, 76 bytes, UDP
```

**Output:**
```
Classification: BENIGN (97.2% confidence)
No threat detected - NTP time synchronization
Analysis stopped at Stage 1 (early exit)
Time: 6 seconds
```

## Architecture

### Multi-Stage Pipeline Flow
```
User Query (Claude Desktop)
    ↓
Tool 1: Validate Input
    ↓
Tool 2: Stage 1 Detection (LightGBM + VAE + GNN)
    ↓
Decision: Benign (38%) → EXIT (6 sec)
         Malicious (62%) → CONTINUE
    ↓
Tool 3: Stage 2 Attack Type (Neural Net)
    ↓
Tool 4: Stage 3 Malware Family (LightGBM)
    ↓
Tool 5: RAG CVE Search (ChromaDB semantic search)
    ↓
Tool 7: Generate Recommendations (CVE-backed)
    ↓
Tool 8: Calculate Risk Score
    ↓
Complete Report (20-25 sec total)
```

### RAG System Architecture
```
Malware Identified: "Mirai"
    ↓
Generate Query: "Mirai IoT telnet vulnerability exploit"
    ↓
Sentence Transformers: Text → 384-dim vector
    ↓
ChromaDB: Cosine similarity search (389 CVEs)
    ↓
Top 10 Matching CVEs (3-5 seconds)
    ↓
Recommendation Engine: CVE-specific actions
```

## Key Innovations

1. **Intelligent Stage Gating** - 38% of traffic exits at Stage 1, saving 15-20 seconds
2. **RAG-Powered Recommendations** - Not rule-based, generated from CVE lakehouse
3. **Multi-Model Integration** - Unified interface for Spark MLlib, PyTorch, ChromaDB
4. **Natural Language Interface** - No API docs needed, plain English
5. **Evidence-Based Intelligence** - Every recommendation references specific CVE

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total Analysis Time | 20-25 sec (malicious), 6 sec (benign) |
| MCP Tools | 11 |
| CVE Database | 389 IoT-specific vulnerabilities |
| RAG Search Time | 3-5 seconds |
| Recommendation Quality | CVE-backed, threat-specific |

## Challenges Solved

### Challenge 1: Model Framework Integration
**Problem:** Spark MLlib, PyTorch, ChromaDB use different formats  
**Solution:** MultiModelLoader class with unified interface

### Challenge 2: RAG Query Quality
**Problem:** Generic queries return irrelevant CVEs  
**Solution:** Malware-specific query generation with context

### Challenge 3: Stage Gating
**Problem:** Need to skip Stages 2-3 for benign without race conditions  
**Solution:** Explicit decision gates in orchestrator

### Challenge 4: Concurrent Requests
**Problem:** Multiple Claude queries at once  
**Solution:** Async/await with proper model lifecycle management

## Files I Created

### Core MCP Implementation (8 files)
- `mcp_server_final.py` (520 lines)
- `multi_stage_orchestrator.py` (380 lines)
- `model_loaders.py` (290 lines)
- `input_validator.py` (150 lines)
- `response_formatter.py` (240 lines)
- `analysis_orchestrator.py` (200 lines)
- `report_generator.py` (180 lines)
- `live_response_formatter.py` (160 lines)

### RAG System (2 files)
- `build_cve_rag.py` (220 lines)
- `download_cve_data.py` (130 lines)

### Testing (3 files)
- `test_comprehensive_analysis.py` (300 lines)
- `test_mcp_manual.py` (180 lines)
- `test_real_samples.py` (210 lines)

**Total:** ~2,960 lines of MCP-specific code

## Integration with Team's Work

- **Phase 1 (Data Engineering):** Uses cleaned IoT-23 dataset from Spark pipeline
- **Phase 2 (ML Models):** Loads trained models (LightGBM, VAE, Neural Net, GNN)
- **My Contribution (Phase 3 MCP):** Makes everything accessible via natural language

## Dependencies on Team Models

My MCP server requires these trained models (created by teammates):
- `models/lightgbm_classifier/` - Stage 1 & 3 models
- `models/vae_binary/` - Stage 1 zero-day detection
- `models/neural_net_multiclass/` - Stage 2 attack classification
- ChromaDB database built from CVE data

## Future Enhancements

- [ ] Add streaming analysis for real-time network traffic
- [ ] Expand CVE database to 1000+ entries
- [ ] Multi-language support (Spanish, Chinese security teams)
- [ ] Integration with SIEM systems (Splunk, ELK)
- [ ] GraphQL API alongside MCP

## References

- Model Context Protocol: https://modelcontextprotocol.io/
- ChromaDB Documentation: https://docs.trychroma.com/
- Sentence Transformers: https://www.sbert.net/

---

**GitHub:** https://github.com/sreekargovind27/malaware_analysis/tree/MCP_final
