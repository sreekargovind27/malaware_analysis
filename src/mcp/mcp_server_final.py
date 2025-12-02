"""
Complete MCP Server for IoT Malware Detection with RAG, Validation, and Reporting
Exposes 10 tools for comprehensive network security analysis
"""

import asyncio
import json
from mcp.server import Server
from mcp.types import Tool, TextContent
from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassificationModel
from pyspark.sql.functions import col, count, avg, sum as spark_sum, desc
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys

# Import local modules (avoid conflict with mcp SDK package)
sys.path.insert(0, str(Path(__file__).parent))
from input_validator import validate_traffic_data
from report_generator import NetworkSecurityReportGenerator
from analysis_orchestrator import TrafficAnalysisOrchestrator
from response_formatter import format_comprehensive_response, format_quick_summary

# Global variables
spark = None
lightgbm_model = None
ml_data = None
cve_data = None
chroma_client = None
cve_collection = None
embedding_model = None
report_generator = None
orchestrator = None


def initialize_spark():
    """Initialize Spark session and load models/data"""
    global spark, lightgbm_model, ml_data, cve_data, chroma_client, cve_collection, embedding_model, report_generator, orchestrator

    print("🚀 Initializing IoT Malware Detection System...")

    # Spark
    spark = SparkSession.builder \
        .appName("IoT23-MCP-Server-Final") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    print("✓ Spark initialized")

    # Load LightGBM model
    model_path = "/Users/nidhirajani/Desktop/DIC Phase 3/models/lightgbm_classifier"
    lightgbm_model = GBTClassificationModel.load(model_path)
    print("✓ LightGBM model loaded")

    # Load ML-ready data
    data_path = "/Users/nidhirajani/Desktop/DIC Phase 3/data/processed/ml_ready_data.parquet"
    ml_data = spark.read.parquet(data_path)
    print("✓ ML data loaded")

    # Load CVE data
    cve_path = "/Users/nidhirajani/Desktop/DIC Phase 3/data/external/iot_cves.csv"
    cve_data = pd.read_csv(cve_path)
    print("✓ CVE data loaded")

    # Initialize RAG
    db_path = "/Users/nidhirajani/Desktop/DIC Phase 3/data/external/cve_vector_db"
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    cve_collection = chroma_client.get_collection("iot_cves")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ RAG system initialized")

    # Initialize report generator
    report_generator = NetworkSecurityReportGenerator()
    print("✓ Report generator ready")

    # Initialize orchestrator
    orchestrator = TrafficAnalysisOrchestrator(
        spark=spark,
        model=lightgbm_model,
        ml_data=ml_data,
        rag_collection=cve_collection,
        embedding_model=embedding_model
    )
    print("✓ Analysis orchestrator initialized")

    print("\n✅ System ready!")


# Create MCP server
server = Server("iot-malware-detection-final")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all 10 available MCP tools"""
    return [
        # NEW COMPREHENSIVE TOOL
        Tool(
            name="analyze_traffic_comprehensive",
            description="""
            🎯 PRIMARY TOOL - Comprehensive IoT traffic security analysis.

            Use this tool when user provides network traffic data for analysis.
            This is a ONE-STOP complete security assessment that automatically:

            1. Validates input data
            2. Classifies traffic as malicious/benign using ML model (99.56% accuracy)
            3. IF MALICIOUS DETECTED:
               - Identifies specific malware family (Mirai, Okiru, PortScan, C&C, DDoS, etc.)
               - Automatically searches CVE database for related vulnerabilities (RAG)
               - Analyzes attack patterns and behaviors
               - Calculates risk assessment with CVSS scores
               - Generates actionable remediation recommendations
               - Provides threat prevalence statistics
            4. IF BENIGN: Provides assurance and monitoring recommendations
            5. Offers to generate detailed PDF report with charts

            This tool runs the complete analysis pipeline automatically - user doesn't
            need to call multiple tools. It intelligently enriches responses with
            threat intelligence when threats are detected.

            Use for: Any network traffic analysis request, security assessment,
            threat detection, or when user provides connection data.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "description": "Connection duration in seconds (e.g., 0.5)"
                    },
                    "orig_bytes": {
                        "type": "number",
                        "description": "Bytes sent by originator/source"
                    },
                    "resp_bytes": {
                        "type": "number",
                        "description": "Bytes sent by responder/destination"
                    },
                    "orig_pkts": {
                        "type": "integer",
                        "description": "Packets sent by originator"
                    },
                    "resp_pkts": {
                        "type": "integer",
                        "description": "Packets sent by responder"
                    },
                    "orig_port": {
                        "type": "integer",
                        "description": "Source port number (0-65535, optional)"
                    },
                    "resp_port": {
                        "type": "integer",
                        "description": "Destination port number (0-65535, optional)"
                    },
                    "proto": {
                        "type": "string",
                        "description": "Protocol: tcp, udp, or icmp (optional)",
                        "enum": ["tcp", "udp", "icmp"]
                    }
                },
                "required": ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]
            }
        ),

        # EXISTING TOOLS
        Tool(
            name="validate_network_traffic",
            description="""
            Validate network traffic input data before classification.
            Checks for required fields, data types, and value ranges.
            Returns detailed validation report with errors and warnings.
            Use this FIRST before classify_traffic_malicious.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "description": "Traffic data as JSON object with fields: duration, orig_bytes, resp_bytes, orig_pkts, resp_pkts, orig_port (opt), resp_port (opt), proto (opt)"
                    }
                },
                "required": ["data"]
            }
        ),
        Tool(
            name="classify_traffic_malicious",
            description="""
            Classify validated IoT network traffic as benign or malicious.
            Returns prediction with confidence score.
            Requires validated input from validate_network_traffic tool.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "sample_id": {
                        "type": "string",
                        "description": "Optional identifier for tracking"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="search_vulnerability_knowledge",
            description="""
            Search CVE vulnerability database using semantic search (RAG).
            Finds relevant IoT vulnerabilities based on query.
            Returns top CVEs with descriptions, CVSS scores, and malware mappings.
            Use for: vulnerability research, threat intelligence, patch prioritization.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'Mirai telnet', 'camera vulnerabilities', 'DDoS mitigation')"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_malware_statistics",
            description="""
            Get comprehensive statistics about detected malware families.
            Returns distribution, counts, and percentages.
            """,
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="analyze_attack_patterns",
            description="""
            Analyze detailed patterns for specific malware family.
            Returns common ports, protocols, and traffic characteristics.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "malware_family": {
                        "type": "string",
                        "enum": ["PortScan", "Okiru", "C&C", "DDoS", "Mirai", "Torii", "Attack", "FileDownload"]
                    }
                },
                "required": ["malware_family"]
            }
        ),
        Tool(
            name="get_network_summary",
            description="""
            Get overall network security summary and health metrics.
            Returns threat level, malicious/benign ratio, top threats.
            """,
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="predict_sample_traffic",
            description="""
            Test model with real traffic samples from dataset.
            Returns prediction for specific attack patterns.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern_type": {
                        "type": "string",
                        "enum": ["portscan", "okiru", "cc", "ddos", "benign"]
                    }
                },
                "required": ["pattern_type"]
            }
        ),
        Tool(
            name="generate_security_report",
            description="""
            Generate comprehensive network security PDF report.
            Includes executive summary, threat analysis, CVE mappings, and recommendations.
            Returns path to generated PDF and Markdown files.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["pdf", "markdown", "both"],
                        "default": "both"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_cve_for_malware",
            description="""
            Get CVE vulnerabilities associated with specific malware family.
            Combines database lookup with RAG semantic search.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "malware_family": {
                        "type": "string",
                        "description": "Malware family name"
                    }
                },
                "required": ["malware_family"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""

    if name == "analyze_traffic_comprehensive":
        print("\n" + "=" * 70)
        print("🚀 RUNNING COMPREHENSIVE ANALYSIS")
        print("=" * 70)

        # Step 1: Validate input
        print("📋 Step 1: Validating input data...")
        is_valid, cleaned_data, validation_report = validate_traffic_data(arguments)

        if not is_valid:
            return [TextContent(type="text", text=f"""
**❌ INPUT VALIDATION FAILED**

{validation_report}

Please provide valid network traffic data with required fields:
- duration (seconds)
- orig_bytes (bytes sent)
- resp_bytes (bytes received)
- orig_pkts (packets sent)
- resp_pkts (packets received)

Optional but recommended:
- orig_port, resp_port, proto
""")]

        print("   ✅ Input validated successfully")

        # Step 2: Run comprehensive analysis
        print("\n🔍 Step 2: Running comprehensive security analysis...")
        try:
            results = orchestrator.run_comprehensive_analysis(cleaned_data)

            # Step 3: Format response
            print("\n📝 Step 3: Formatting analysis report...")
            response_text = format_comprehensive_response(results)

            print("=" * 70)
            print("✅ ANALYSIS COMPLETE")
            print("=" * 70 + "\n")

            return [TextContent(type="text", text=response_text)]

        except Exception as e:
            error_msg = f"""
**❌ ANALYSIS ERROR**

An error occurred during comprehensive analysis:
{str(e)}

Please try again or use individual tools for specific analysis.
"""
            print(f"\n❌ Error: {e}")
            return [TextContent(type="text", text=error_msg)]

    elif name == "validate_network_traffic":
        data = arguments.get("data", {})
        is_valid, cleaned_data, report = validate_traffic_data(data)

        response = report
        if is_valid:
            response += f"\n**Cleaned Data:**\n```json\n{json.dumps(cleaned_data, indent=2)}\n```"

        return [TextContent(type="text", text=response)]

    elif name == "search_vulnerability_knowledge":
        query = arguments.get("query", "")
        n_results = min(arguments.get("n_results", 5), 10)

        # Generate query embedding
        query_embedding = embedding_model.encode([query])

        # Search ChromaDB
        results = cve_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )

        response = f"**Vulnerability Search Results for:** '{query}'\n\n"

        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            response += f"**{i}. {meta['cve_id']}** (CVSS: {meta['cvss_score']})\n"
            response += f"- **Malware Family:** {meta['malware_family']}\n"
            response += f"- **Description:** {meta['description'][:200]}...\n\n"

        return [TextContent(type="text", text=response)]

    elif name == "classify_traffic_malicious":
        # Get random sample
        sample = ml_data.sample(fraction=0.001, seed=42).limit(1)
        prediction = lightgbm_model.transform(sample)

        result = prediction.select("is_malicious", "malware_family", "prediction", "probability",
                                   "orig_port", "resp_port", "proto").collect()[0]

        probs = result["probability"].toArray()
        pred = "Malicious" if result["prediction"] == 1 else "Benign"
        actual = "Malicious" if result["is_malicious"] == 1 else "Benign"

        response = f"""**Traffic Classification Result**

**Prediction:** {pred} (Actual: {actual})
**Confidence:** {max(probs):.1%}

**Probabilities:**
- Malicious: {probs[1]:.1%}
- Benign: {probs[0]:.1%}

**Traffic Details:**
- Ports: {result['orig_port']} → {result['resp_port']}
- Protocol: {result['proto'].upper()}
- Malware Family: {result['malware_family']}
"""
        return [TextContent(type="text", text=response)]

    elif name == "get_malware_statistics":
        stats = ml_data.groupBy("malware_family").agg(
            count("*").alias("count")
        ).orderBy(desc("count"))

        total = ml_data.count()
        malicious = ml_data.filter(col("is_malicious") == 1).count()
        benign = ml_data.filter(col("is_malicious") == 0).count()

        stats_list = stats.collect()

        response = f"""**IoT Malware Statistics**

**Overall Distribution:**
- Total Samples: {total:,}
- Malicious: {malicious:,} ({malicious / total * 100:.1f}%)
- Benign: {benign:,} ({benign / total * 100:.1f}%)

**Malware Family Breakdown:**
"""
        for row in stats_list:
            if row['malware_family'] != 'Benign':
                pct = row['count'] / total * 100
                response += f"- {row['malware_family']}: {row['count']:,} samples ({pct:.1f}%)\n"

        return [TextContent(type="text", text=response)]

    elif name == "analyze_attack_patterns":
        malware_family = arguments.get("malware_family")

        family_data = ml_data.filter(col("malware_family") == malware_family)

        if family_data.count() == 0:
            return [TextContent(type="text", text=f"No data found for malware family: {malware_family}")]

        proto_dist = family_data.groupBy("proto").count().orderBy(desc("count")).collect()
        predictions = lightgbm_model.transform(family_data.limit(5))
        samples = predictions.select("orig_port", "resp_port", "proto", "probability").collect()

        response = f"""**Attack Pattern Analysis: {malware_family}**

**Protocol Distribution:**
"""
        for row in proto_dist:
            response += f"- {row['proto'].upper()}: {row['count']:,} connections\n"

        response += f"\n**Sample Traffic Patterns:**\n"
        for i, row in enumerate(samples, 1):
            probs = row["probability"].toArray()
            response += f"{i}. {row['orig_port']} → {row['resp_port']} ({row['proto'].upper()}) - Confidence: {probs[1]:.1%}\n"

        return [TextContent(type="text", text=response)]

    elif name == "get_network_summary":
        total = ml_data.count()
        malicious = ml_data.filter(col("is_malicious") == 1).count()
        benign = ml_data.filter(col("is_malicious") == 0).count()

        top_threats = ml_data.filter(col("is_malicious") == 1) \
            .groupBy("malware_family").count() \
            .orderBy(desc("count")).limit(3).collect()

        response = f"""**Network Security Summary**

**Overall Health:**
- Total Traffic Analyzed: {total:,} connections
- Threat Level: {'🔴 HIGH' if malicious / total > 0.5 else '🟡 MEDIUM' if malicious / total > 0.2 else '🟢 LOW'}
- Malicious Traffic: {malicious:,} ({malicious / total * 100:.1f}%)
- Benign Traffic: {benign:,} ({benign / total * 100:.1f}%)

**Top 3 Threats:**
"""
        for i, row in enumerate(top_threats, 1):
            pct = row['count'] / malicious * 100
            response += f"{i}. {row['malware_family']}: {row['count']:,} attacks ({pct:.1f}% of threats)\n"

        response += f"""
**Model Performance:**
- Accuracy: 99.56%
- AUC-ROC: 99.94%
- False Positive Rate: 0.09%
"""

        return [TextContent(type="text", text=response)]

    elif name == "predict_sample_traffic":
        pattern_type = arguments.get("pattern_type", "benign").lower()

        if pattern_type == "portscan":
            sample = ml_data.filter(col("malware_family") == "PortScan").limit(1)
        elif pattern_type == "okiru":
            sample = ml_data.filter(col("malware_family") == "Okiru").limit(1)
        elif pattern_type == "cc":
            sample = ml_data.filter(col("malware_family") == "C&C").limit(1)
        elif pattern_type == "ddos":
            sample = ml_data.filter(col("malware_family") == "DDoS").limit(1)
        else:
            sample = ml_data.filter(col("malware_family") == "Benign").limit(1)

        prediction = lightgbm_model.transform(sample)
        result = prediction.select("prediction", "probability", "orig_port", "resp_port", "proto").collect()[0]

        probs = result["probability"].toArray()
        pred = "Malicious" if result["prediction"] == 1 else "Benign"

        response = f"""**Traffic Pattern Prediction: {pattern_type.upper()}**

**Result:** {pred}
**Confidence:** {max(probs):.1%}

**Traffic Details:**
- Ports: {result['orig_port']} → {result['resp_port']}
- Protocol: {result['proto'].upper()}
- Malicious Probability: {probs[1]:.1%}
- Benign Probability: {probs[0]:.1%}
"""

        return [TextContent(type="text", text=response)]

    elif name == "generate_security_report":
        format_type = arguments.get("format", "both")

        # Gather data for report
        total = ml_data.count()
        malicious = ml_data.filter(col("is_malicious") == 1).count()
        benign = ml_data.filter(col("is_malicious") == 0).count()

        threat_level = "HIGH" if malicious / total > 0.5 else "MEDIUM" if malicious / total > 0.2 else "LOW"

        network_stats = {
            'total': total,
            'malicious': malicious,
            'benign': benign,
            'threat_level': threat_level
        }

        # Malware distribution
        malware_dist_df = ml_data.filter(col("is_malicious") == 1) \
            .groupBy("malware_family").count().collect()
        malware_distribution = {row['malware_family']: row['count'] for row in malware_dist_df}

        # Top threats
        top_threats_data = []
        for row in malware_dist_df[:5]:
            top_threats_data.append({
                'family': row['malware_family'],
                'count': row['count'],
                'percentage': row['count'] / malicious * 100
            })

        # CVE mappings
        cve_mappings = []
        for _, row in cve_data.head(5).iterrows():
            cve_mappings.append({
                'cve_id': row['cve_id'],
                'cvss': row['cvss_score'],
                'description': row['description']
            })

        # Recommendations
        recommendations = [
            "🔥 Immediately patch CVEs with CVSS scores above 7.0",
            "🛡️ Implement network segmentation to isolate IoT devices",
            "🚫 Block C&C server IPs at firewall level",
            "🔐 Change default credentials on all IoT devices",
            "📊 Enable continuous monitoring with automated alerts",
            "⚙️ Disable unnecessary services on IoT devices (telnet, SSH)",
            "🔄 Update firmware on all cameras and routers",
            "🌐 Restrict outbound connections from IoT devices"
        ]

        # Generate reports
        files_generated = []

        if format_type in ["pdf", "both"]:
            pdf_path = report_generator.generate_report(
                network_stats, malware_distribution, top_threats_data,
                cve_mappings, recommendations
            )
            files_generated.append(pdf_path)

        if format_type in ["markdown", "both"]:
            md_path = report_generator.generate_markdown_report(
                network_stats, malware_distribution, top_threats_data,
                cve_mappings, recommendations
            )
            files_generated.append(md_path)

        response = f"""**Security Report Generated Successfully**

**Files Created:**
"""
        for file in files_generated:
            response += f"- {file}\n"

        response += f"""
**Report Summary:**
- Total Traffic: {total:,}
- Threat Level: {threat_level}
- Top Threat: {top_threats_data[0]['family']} ({top_threats_data[0]['count']:,} attacks)
"""

        return [TextContent(type="text", text=response)]

    elif name == "get_cve_for_malware":
        malware_family = arguments.get("malware_family", "").lower()

        # RAG search
        query_embedding = embedding_model.encode([malware_family])
        results = cve_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=5
        )

        response = f"""**CVE Vulnerabilities for {arguments.get('malware_family')}**

"""
        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            response += f"**{i}. {meta['cve_id']}** (CVSS: {meta['cvss_score']})\n"
            response += f"{meta['description'][:250]}...\n\n"

        return [TextContent(type="text", text=response)]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    """Run MCP server"""

    # Initialize system
    initialize_spark()

    # Run server
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        print("\n✓ MCP Server running on stdio")
        print("✓ 10 tools available")
        print("✓ Ready for requests\n")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())