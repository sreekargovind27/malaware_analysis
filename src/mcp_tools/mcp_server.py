"""
MCP Server for IoT Malware Detection
Exposes ML models and analytics as tools for Claude
"""

import asyncio
import json
from mcp.server import Server
from mcp.types import Tool, TextContent
from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassificationModel
import os

# Global variables
spark = None
lightgbm_model = None


def initialize_spark():
    """Initialize Spark session and load models"""
    global spark, lightgbm_model

    spark = SparkSession.builder \
        .appName("IoT23-MCP-Server") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    # Load LightGBM model
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(project_root, "models", "lightgbm_classifier")

    lightgbm_model = GBTClassificationModel.load(model_path)

    print("✓ Spark initialized")
    print("✓ LightGBM model loaded")


def classify_traffic(features_dict):
    """
    Classify network traffic as benign or malicious

    Args:
        features_dict: Dictionary with traffic features

    Returns:
        Dictionary with prediction and confidence
    """
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import Row

    # Extract features in correct order
    feature_order = [
        "duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts",
        "orig_ip_bytes", "resp_ip_bytes", "total_bytes", "total_packets",
        "bytes_per_packet", "resp_orig_ratio", "packets_per_second",
        "is_telnet_port", "is_ssh_port", "is_http_port", "proto_encoded",
        "hour_of_day", "day_of_week", "orig_port", "resp_port"
    ]

    # Build feature vector
    feature_values = [float(features_dict.get(f, 0)) for f in feature_order]
    features = Vectors.dense(feature_values)

    # Create DataFrame
    data = [Row(features=features)]
    df = spark.createDataFrame(data)

    # Make prediction
    prediction_df = lightgbm_model.transform(df)
    result = prediction_df.select("prediction", "probability").collect()[0]

    prediction = int(result["prediction"])
    probabilities = result["probability"].toArray().tolist()

    return {
        "prediction": "Malicious" if prediction == 1 else "Benign",
        "confidence": max(probabilities),
        "malicious_probability": probabilities[1],
        "benign_probability": probabilities[0]
    }


# Create MCP server
server = Server("iot-malware-detection")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="classify_traffic_malicious",
            description="""
            Classify network traffic as benign or malicious using LightGBM model.

            This tool analyzes IoT network traffic patterns to detect malware infections.
            It uses a gradient boosting model trained on 70,000+ real IoT malware samples
            from the IoT-23 dataset (Mirai, Torii, Okiru botnets).

            Input: Network flow features (duration, bytes, packets, ports, protocol)
            Output: Classification (Benign/Malicious) with confidence score

            Use this for:
            - Real-time traffic analysis
            - Incident response investigations  
            - IoT device security monitoring
            - Malware detection in network logs
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "duration": {"type": "number", "description": "Connection duration in seconds"},
                    "orig_bytes": {"type": "number", "description": "Bytes sent by originator"},
                    "resp_bytes": {"type": "number", "description": "Bytes sent by responder"},
                    "orig_pkts": {"type": "number", "description": "Packets sent by originator"},
                    "resp_pkts": {"type": "number", "description": "Packets sent by responder"},
                    "orig_ip_bytes": {"type": "number", "description": "IP-level bytes from originator"},
                    "resp_ip_bytes": {"type": "number", "description": "IP-level bytes from responder"},
                    "orig_port": {"type": "integer", "description": "Source port number"},
                    "resp_port": {"type": "integer", "description": "Destination port number"},
                    "proto": {"type": "string", "description": "Protocol: tcp, udp, or icmp"},
                    "hour_of_day": {"type": "integer", "description": "Hour (0-23), optional"},
                    "day_of_week": {"type": "integer", "description": "Day (1-7), optional"}
                },
                "required": ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""

    if name == "classify_traffic_malicious":
        # Calculate derived features
        args = arguments.copy()

        # Set defaults
        args.setdefault("orig_ip_bytes", args.get("orig_bytes", 0))
        args.setdefault("resp_ip_bytes", args.get("resp_bytes", 0))
        args.setdefault("orig_port", 0)
        args.setdefault("resp_port", 0)
        args.setdefault("hour_of_day", 12)
        args.setdefault("day_of_week", 1)

        # Derived features
        args["total_bytes"] = args["orig_bytes"] + args["resp_bytes"]
        args["total_packets"] = args["orig_pkts"] + args["resp_pkts"]
        args["bytes_per_packet"] = args["total_bytes"] / args["total_packets"] if args["total_packets"] > 0 else 0
        args["resp_orig_ratio"] = args["resp_bytes"] / args["orig_bytes"] if args["orig_bytes"] > 0 else 0
        args["packets_per_second"] = args["total_packets"] / args["duration"] if args["duration"] > 0 else 0

        # Port indicators
        args["is_telnet_port"] = 1 if args["orig_port"] == 23 or args["resp_port"] == 23 else 0
        args["is_ssh_port"] = 1 if args["orig_port"] == 22 or args["resp_port"] == 22 else 0
        args["is_http_port"] = 1 if args["orig_port"] == 80 or args["resp_port"] == 80 else 0

        # Protocol encoding
        proto = args.get("proto", "").lower()
        args["proto_encoded"] = 1 if proto == "tcp" else (2 if proto == "udp" else (3 if proto == "icmp" else 0))

        # Classify
        result = classify_traffic(args)

        # Format response
        response = f"""**Traffic Classification Result**

**Prediction:** {result['prediction']}
**Confidence:** {result['confidence']:.2%}

**Probability Breakdown:**
- Malicious: {result['malicious_probability']:.2%}
- Benign: {result['benign_probability']:.2%}

**Traffic Summary:**
- Duration: {args['duration']:.2f}s
- Total Bytes: {args['total_bytes']:,}
- Total Packets: {args['total_packets']}
- Protocol: {args.get('proto', 'unknown').upper()}
- Ports: {args['orig_port']} → {args['resp_port']}
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