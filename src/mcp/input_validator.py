"""
Input Validation for Network Traffic Data
Handles messy real-world input with helpful error messages
"""

from typing import Dict, Tuple, List, Optional
import re
import json


class NetworkTrafficValidator:
    """Validates network traffic input data"""

    # Field requirements
    MANDATORY_FIELDS = {
        'duration': {'type': float, 'min': 0, 'description': 'Connection duration in seconds'},
        'orig_bytes': {'type': int, 'min': 0, 'description': 'Bytes sent by originator'},
        'resp_bytes': {'type': int, 'min': 0, 'description': 'Bytes sent by responder'},
        'orig_pkts': {'type': int, 'min': 0, 'description': 'Packets sent by originator'},
        'resp_pkts': {'type': int, 'min': 0, 'description': 'Packets sent by responder'}
    }

    OPTIONAL_FIELDS = {
        'orig_port': {'type': int, 'min': 0, 'max': 65535, 'description': 'Originator port'},
        'resp_port': {'type': int, 'min': 0, 'max': 65535, 'description': 'Responder port'},
        'proto': {'type': str, 'values': ['tcp', 'udp', 'icmp'], 'description': 'Protocol'}
    }

    def __init__(self):
        self.warnings = []
        self.errors = []

    def validate(self, data: Dict) -> Tuple[bool, Dict, List[str], List[str]]:
        """
        Validate input data

        Returns:
            (is_valid, cleaned_data, errors, warnings)
        """
        self.warnings = []
        self.errors = []
        cleaned_data = {}

        # Check mandatory fields
        for field, specs in self.MANDATORY_FIELDS.items():
            if field not in data or data[field] is None:
                self.errors.append(f"❌ Missing required field: {field} ({specs['description']})")
                continue

            # Validate type and range
            try:
                value = specs['type'](data[field])
                if value < specs['min']:
                    self.errors.append(f"❌ {field} must be >= {specs['min']}, got {value}")
                else:
                    cleaned_data[field] = value
            except (ValueError, TypeError):
                self.errors.append(f"❌ {field} must be {specs['type'].__name__}, got {type(data[field]).__name__}")

        # Check optional fields
        for field, specs in self.OPTIONAL_FIELDS.items():
            if field in data and data[field] is not None:
                try:
                    if specs['type'] == str:
                        value = str(data[field]).lower()
                        if 'values' in specs and value not in specs['values']:
                            self.warnings.append(
                                f"⚠️  {field} should be one of {specs['values']}, got '{value}' - using default")
                            cleaned_data[field] = 'tcp'  # Default
                        else:
                            cleaned_data[field] = value
                    else:
                        value = specs['type'](data[field])
                        if 'min' in specs and value < specs['min']:
                            self.warnings.append(f"⚠️  {field} out of range (min={specs['min']}), got {value}")
                        elif 'max' in specs and value > specs['max']:
                            self.warnings.append(f"⚠️  {field} out of range (max={specs['max']}), got {value}")
                        else:
                            cleaned_data[field] = value
                except (ValueError, TypeError):
                    self.warnings.append(f"⚠️  {field} has invalid type, expected {specs['type'].__name__}")
            else:
                self.warnings.append(f"⚠️  Optional field missing: {field} - prediction confidence may be lower")

        is_valid = len(self.errors) == 0

        return is_valid, cleaned_data, self.errors, self.warnings

    def parse_csv_string(self, csv_string: str) -> Dict:
        """Parse CSV format: duration,orig_bytes,resp_bytes,orig_pkts,resp_pkts,orig_port,resp_port,proto"""
        parts = csv_string.strip().split(',')

        if len(parts) < 5:
            raise ValueError("CSV must have at least 5 fields: duration,orig_bytes,resp_bytes,orig_pkts,resp_pkts")

        data = {
            'duration': parts[0].strip(),
            'orig_bytes': parts[1].strip(),
            'resp_bytes': parts[2].strip(),
            'orig_pkts': parts[3].strip(),
            'resp_pkts': parts[4].strip()
        }

        if len(parts) > 5:
            data['orig_port'] = parts[5].strip()
        if len(parts) > 6:
            data['resp_port'] = parts[6].strip()
        if len(parts) > 7:
            data['proto'] = parts[7].strip()

        return data

    def parse_natural_language(self, text: str) -> Dict:
        """Parse natural language input (best effort)"""
        data = {}

        # Extract numbers with units
        duration_match = re.search(r'(\d+\.?\d*)\s*(second|sec|s)', text, re.I)
        if duration_match:
            data['duration'] = duration_match.group(1)

        bytes_sent = re.search(r'(\d+)\s*bytes?\s*sent', text, re.I)
        if bytes_sent:
            data['orig_bytes'] = bytes_sent.group(1)

        bytes_recv = re.search(r'(\d+)\s*bytes?\s*(received|recv)', text, re.I)
        if bytes_recv:
            data['resp_bytes'] = bytes_recv.group(1)

        pkts_sent = re.search(r'(\d+)\s*packets?\s*sent', text, re.I)
        if pkts_sent:
            data['orig_pkts'] = pkts_sent.group(1)

        pkts_recv = re.search(r'(\d+)\s*packets?\s*(received|recv)', text, re.I)
        if pkts_recv:
            data['resp_pkts'] = pkts_recv.group(1)

        # Extract ports
        port_match = re.search(r'port\s*(\d+)', text, re.I)
        if port_match:
            data['orig_port'] = port_match.group(1)

        # Extract protocol
        if 'tcp' in text.lower():
            data['proto'] = 'tcp'
        elif 'udp' in text.lower():
            data['proto'] = 'udp'
        elif 'icmp' in text.lower():
            data['proto'] = 'icmp'

        return data

    def format_validation_report(self, is_valid: bool, errors: List[str], warnings: List[str]) -> str:
        """Format validation results as human-readable report"""
        report = "**Input Validation Report**\n\n"

        if is_valid:
            report += "✅ **Status:** Valid - Ready for prediction\n\n"
        else:
            report += "❌ **Status:** Invalid - Cannot predict\n\n"

        if errors:
            report += "**Errors:**\n"
            for error in errors:
                report += f"{error}\n"
            report += "\n"

        if warnings:
            report += "**Warnings:**\n"
            for warning in warnings:
                report += f"{warning}\n"
            report += "\n"

        if not is_valid:
            report += "**Required Fields:**\n"
            for field, specs in self.MANDATORY_FIELDS.items():
                report += f"- {field}: {specs['description']}\n"

        return report


# Helper function for MCP server
def validate_traffic_data(data: Dict) -> Tuple[bool, Dict, str]:
    """
    Convenience function for MCP server

    Returns:
        (is_valid, cleaned_data, report_text)
    """
    validator = NetworkTrafficValidator()

    # Auto-detect format if string
    if isinstance(data, str):
        if data.strip().startswith('{'):
            data = json.loads(data)
        elif ',' in data:
            data = validator.parse_csv_string(data)
        else:
            data = validator.parse_natural_language(data)

    is_valid, cleaned_data, errors, warnings = validator.validate(data)
    report = validator.format_validation_report(is_valid, errors, warnings)

    return is_valid, cleaned_data, report