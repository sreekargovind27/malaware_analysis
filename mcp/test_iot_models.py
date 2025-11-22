"""
test_iot_models.py

Standalone sanity tests for the IoT ThreatEngine and wrapper functions.

Usage:
    cd /path/to/dic_phase2/mcp
    python test_iot_models.py
    python test_iot_models.py --csv /path/to/engineered_flows.csv
"""

import os
import sys
import argparse
import json
from pprint import pprint

# Ensure project root is on the path so `mcp.tools` or `tools` can import.
current_file = os.path.abspath(__file__)
mcp_dir = os.path.dirname(current_file)
project_root = os.path.dirname(mcp_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Prefer `mcp.tools`, fall back to plain `tools.py`
try:
    from mcp.tools import (
        engine,
        scan_flow,
        scan_file,
        lookup_malware_info,
        get_mitigation_plan,
        check_device_history,
        get_model_performance,
        explain_prediction,
    )

    IMPORT_SOURCE = "mcp.tools"
except Exception:
    from tools import (
        engine,
        scan_flow,
        scan_file,
        lookup_malware_info,
        get_mitigation_plan,
        check_device_history,
        get_model_performance,
        explain_prediction,
    )

    IMPORT_SOURCE = "tools.py"


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_model_registry() -> None:
    """Check that the registry sees the models and paths correctly."""
    print_header(f"[1] Import source: {IMPORT_SOURCE}")
    print("Project root:", project_root)
    print("MCP dir     :", mcp_dir)

    print_header("[2] Model registry / get_model_performance()")
    stats = get_model_performance()
    print(stats)


def safe_parse_result(result_str: str):
    """Best-effort convert the scan_flow string result back to a dict for nice printing."""
    if not isinstance(result_str, str):
        return result_str
    try:
        clean = result_str.replace("'", '"')
        return json.loads(clean)
    except Exception:
        return result_str


def test_scan_flow_samples() -> None:
    """Run a few synthetic flows through scan_flow()."""
    print_header("[3] Testing scan_flow() on sample flows")

    sample_flows = [
        {"name": "Likely benign HTTP", "duration": 1.2, "bytes_t": 1500, "port": 80, "proto": "tcp"},
        {"name": "Large Telnet - possible Mirai", "duration": 10.5, "bytes_t": 90000, "port": 23, "proto": "tcp"},
        {"name": "DNS traffic", "duration": 0.2, "bytes_t": 600, "port": 53, "proto": "udp"},
        {"name": "Random high port burst", "duration": 3.0, "bytes_t": 500000, "port": 5555, "proto": "tcp"},
    ]

    for flow in sample_flows:
        print("\n---", flow["name"], "---")
        try:
            result_str = scan_flow(
                duration=flow["duration"],
                bytes_t=flow["bytes_t"],
                port=flow["port"],
                proto=flow["proto"],
            )
            parsed = safe_parse_result(result_str)
            pprint(parsed)
        except Exception as e:
            print(f"❌ scan_flow() raised an error: {e}")


def test_csv_scan(csv_path: str) -> None:
    """Test scan_file() against a CSV of flows."""
    print_header(f"[4] Testing scan_file() on CSV: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"❌ CSV file does not exist: {csv_path}")
        return

    try:
        result = scan_file(csv_path)
        print("scan_file() result:")
        print(result)
    except Exception as e:
        print(f"❌ scan_file() raised an error: {e}")


def test_helpers() -> None:
    """Check malware info, mitigation, device history, and explainer."""
    print_header("[5] Testing helper functions")

    queries = ["Mirai", "Torii", "randomstuff"]
    for q in queries:
        print(f"\nQuery: {q}")
        try:
            info = lookup_malware_info(q)
            mit = get_mitigation_plan(q)
            print("lookup_malware_info ->", info)
            print("get_mitigation_plan ->", mit)
        except Exception as e:
            print(f"❌ Helper functions failed for '{q}': {e}")

    print("\nTesting check_device_history() and explain_prediction():")
    try:
        print("check_device_history(192.168.1.10) ->", check_device_history("192.168.1.10"))
    except Exception as e:
        print("❌ check_device_history error:", e)

    try:
        print("explain_prediction('flow-123') ->", explain_prediction("flow-123"))
    except Exception as e:
        print("❌ explain_prediction error:", e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity tests for IoT ThreatEngine models & tool functions."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to a CSV with flows to test scan_file().",
    )
    args = parser.parse_args()

    test_model_registry()
    test_scan_flow_samples()

    if args.csv:
        test_csv_scan(args.csv)

    test_helpers()

    print_header("DONE")


if __name__ == "__main__":
    main()
