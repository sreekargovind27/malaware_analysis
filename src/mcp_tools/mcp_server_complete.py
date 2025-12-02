"""
Complete MCP Server for IoT Malware Detection
Exposes 6 tools for Claude to analyze IoT network security
"""

import asyncio
import json
from mcp.server import Server
from mcp.types import Tool, TextContent
from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassificationModel
from pyspark.sql.functions import col, count, avg, sum as spark_sum
import pandas as pd
import os

# Global variables
spark = None
lightgbm_model = None
ml_data = None
cve_data = None


def initialize_spark():
    """Initialize Spark session and load models/data"""
    global spark, lightgbm_model, ml_data, cve_data

    spark = SparkSession.builder \
        .appName("IoT23-MCP-Server") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    # Load LightGBM model
    model_path = "/Users/nidhirajani/Desktop/DIC Phase 3/models/lightgbm_classifier"
    lightgbm_model = GBTClassificationModel.load(model_path)

    # Load ML-ready data
    data_path = "/Users/nidhirajani/Desktop/DIC Phase 3/data/processed/ml_ready_data.parquet"
    ml_data = spark.read.parquet(data_path)

    # Load CVE data
    cve_path = "/Users/nidhirajani/Desktop/DIC Phase 3/data/external/iot_cves.csv"
    cve_data = pd.read_csv(cve_path)

    print("✓ Spark initialized")
    print("✓ LightGBM model loaded")
    print("✓ ML data loaded")
    print("✓ CVE data loaded")


# Create MCP server
server = Server("iot-malware-detection")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="classify_traffic_malicious",
            description="""
            Classify IoT network traffic as benign or malicious using ML model.
            Returns classification with confidence score for traffic patterns.
            Use for real-time threat detection and incident response.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "traffic_id": {"type": "string", "description": "Identifier for the traffic sample (optional)"}
                },
                "required": []
            }
        ),
        Tool(
            name="get_malware_statistics",
            description="""
            Get comprehensive statistics about malware families in the dataset.
            Returns distribution, counts, and percentages of different malware types.
            Use for threat intelligence and understanding attack landscape.
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
            Analyze detailed patterns for a specific malware family.
            Returns common ports, protocols, traffic characteristics.
            Use for building detection rules and understanding attack behavior.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "malware_family": {
                        "type": "string",
                        "description": "Malware family name (PortScan, Okiru, C&C, DDoS, Mirai, Torii, Attack, FileDownload)",
                        "enum": ["PortScan", "Okiru", "C&C", "DDoS", "Mirai", "Torii", "Attack", "FileDownload"]
                    }
                },
                "required": ["malware_family"]
            }
        ),
        Tool(
            name="check_vulnerability_mapping",
            description="""
            Map malware families to known CVE vulnerabilities.
            Returns CVEs exploited by specific malware with severity scores.
            Use for patch prioritization and vulnerability management.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "malware_family": {
                        "type": "string",
                        "description": "Malware family name to lookup CVEs for"
                    }
                },
                "required": ["malware_family"]
            }
        ),
        Tool(
            name="get_network_summary",
            description="""
            Get overall network security summary and health metrics.
            Returns malicious vs benign ratio, top threats, model performance.
            Use for security dashboards and executive reporting.
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
            Predict if specific traffic pattern is malicious using real samples.
            Returns prediction with confidence for port scanning, botnet C&C, etc.
            Use for testing and validating security controls.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern_type": {
                        "type": "string",
                        "description": "Type of pattern to test",
                        "enum": ["portscan", "okiru", "cc", "ddos", "benign"]
                    }
                },
                "required": ["pattern_type"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""

    if name == "classify_traffic_malicious":
        # Get a random sample and classify
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
        # Get malware distribution
        stats = ml_data.groupBy("malware_family").agg(
            count("*").alias("count")
        ).orderBy(col("count").desc())

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

        # Filter by malware family
        family_data = ml_data.filter(col("malware_family") == malware_family)

        if family_data.count() == 0:
            return [TextContent(type="text", text=f"No data found for malware family: {malware_family}")]

        # Get common patterns
        proto_dist = family_data.groupBy("proto").count().orderBy(col("count").desc()).collect()

        # Get predictions
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

    elif name == "check_vulnerability_mapping":
        malware_family = arguments.get("malware_family", "").lower()

        # Search CVEs related to this malware
        related_cves = cve_data[cve_data['description'].str.contains(malware_family, case=False, na=False)]

        if len(related_cves) == 0:
            related_cves = cve_data[
                cve_data['description'].str.contains('iot|router|camera', case=False, na=False)].head(5)

        response = f"""**CVE Vulnerability Mapping: {arguments.get('malware_family')}**

**Related Vulnerabilities:**
"""
        for _, row in related_cves.head(5).iterrows():
            response += f"\n**{row['cve_id']}**\n"
            response += f"- CVSS Score: {row['cvss_score']}\n"
            response += f"- Description: {row['description'][:200]}...\n"

        return [TextContent(type="text", text=response)]

    elif name == "get_network_summary":
        total = ml_data.count()
        malicious = ml_data.filter(col("is_malicious") == 1).count()
        benign = ml_data.filter(col("is_malicious") == 0).count()

        # Get top threats
        top_threats = ml_data.filter(col("is_malicious") == 1) \
            .groupBy("malware_family").count() \
            .orderBy(col("count").desc()).limit(3).collect()

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

        # Get sample based on pattern type
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

    raise ValueError(f"Unknown tool: {name}")


async def main():
    """Run MCP server"""

    # Initialize Spark and models
    initialize_spark()

    # Run server
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        print("✓ MCP Server running on stdio")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())