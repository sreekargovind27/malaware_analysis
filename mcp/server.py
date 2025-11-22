import sys
import os

current_file_path = os.path.abspath(__file__)
mcp_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(mcp_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastmcp import FastMCP

# Import tools + wrappers
from tools import (
    engine,
    scan_flow as scan_flow_impl,
    scan_file as scan_file_impl,
    scan_feature_vector as scan_feature_vector_impl,
    lookup_malware_info as lookup_malware_info_impl,
    get_mitigation_plan as get_mitigation_plan_impl,
    check_device_history as check_device_history_impl,
    get_model_performance as get_model_performance_impl,
    explain_prediction as explain_prediction_impl,
)

mcp = FastMCP("IoT-23-Threat-Defense")


@mcp.tool()
def scan_flow(
        duration: float,
        bytes_t: int,
        port: int,
        proto: str = "tcp",
) -> str:
    """
    Analyze a single flow using minimal fields (duration, bytes, port, proto).

    NOTE:
    - This is NOT full-feature Option B.
    - For true model-accurate inference, use `scan_feature_vector` with a full
      engineered feature dictionary, or upload a CSV and use `scan_file`.
    """
    return scan_flow_impl(duration, bytes_t, port, proto)


@mcp.tool()
def scan_file(file_path: str) -> str:
    """
    Analyze a CSV file of flows using the FULL numeric feature set (Option B).

    Requirements:
    - The CSV must contain all numeric feature columns used during training
      (those from feature_list.joblib, minus the 8 non-numeric ones).
    """
    return scan_file_impl(file_path)


@mcp.tool()
def scan_feature_vector(features: dict) -> str:
    """
    Analyze a single flow using a FULL engineered feature dictionary (Option B).

    `features` must contain (at least) all of the numeric features used during
    training. Missing features are treated as 0.

    Example (keys shortened):
    {
        "id.orig_p": 52000,
        "id.resp_p": 23,
        "orig_bytes": 1234,
        "resp_bytes": 5678,
        "duration": 10.5,
        "upload_ratio": 0.8,
        "proto_idx": 0,
        ...
    }
    """
    return scan_feature_vector_impl(features)


@mcp.tool()
def lookup_malware_info(malware_name: str) -> str:
    """Look up information on a known IoT malware family (Mirai, Torii, etc.)."""
    return lookup_malware_info_impl(malware_name)


@mcp.tool()
def get_mitigation_plan(threat_name: str) -> str:
    """Return mitigation guidance for a given threat or attack family name."""
    return get_mitigation_plan_impl(threat_name)


@mcp.tool()
def check_device_history(ip_address: str) -> str:
    """Return a pseudo-history for a device IP (clean / spikes / repeated attacks)."""
    return check_device_history_impl(ip_address)


@mcp.tool()
def get_model_performance() -> str:
    """Return a registry of which models are currently available and on disk."""
    return get_model_performance_impl()


@mcp.tool()
def explain_prediction(flow_id: str) -> str:
    """Explain why a given flow (by id) was flagged (placeholder explainer)."""
    return explain_prediction_impl(flow_id)


if __name__ == "__main__":
    # Start MCP server for Claude Desktop
    mcp.run()
