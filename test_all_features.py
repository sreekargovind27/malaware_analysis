"""
Test all 3 new features
"""

import sys
from pathlib import Path

# Get absolute paths
project_root = Path("/Users/nidhirajani/Desktop/DIC Phase 3")
sys.path.insert(0, str(project_root / "src" / "mcp"))
sys.path.insert(0, str(project_root / "src" / "utils"))

from input_validator import NetworkTrafficValidator
from build_cve_rag import build_cve_vector_db
from report_generator import NetworkSecurityReportGenerator
import chromadb
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("TESTING ALL 3 FEATURES")
print("=" * 60)

# TEST 1: Input Validation
print("\n1️⃣  TESTING INPUT VALIDATION")
print("-" * 60)

validator = NetworkTrafficValidator()

# Test complete data
good_data = {
    'duration': 0.5,
    'orig_bytes': 100,
    'resp_bytes': 200,
    'orig_pkts': 5,
    'resp_pkts': 10,
    'orig_port': 80,
    'resp_port': 443,
    'proto': 'tcp'
}

is_valid, cleaned, errors, warnings = validator.validate(good_data)
print(f"✅ Complete data: {'VALID' if is_valid else 'INVALID'}")
print(f"   Errors: {len(errors)}, Warnings: {len(warnings)}")

# Test incomplete data
bad_data = {
    'duration': 0.5,
    'orig_bytes': 100
    # Missing required fields
}

is_valid, cleaned, errors, warnings = validator.validate(bad_data)
print(f"❌ Incomplete data: {'VALID' if is_valid else 'INVALID'}")
print(f"   Errors: {len(errors)}, Warnings: {len(warnings)}")

# Test CSV parsing
csv_string = "0.5,100,200,5,10,80,443,tcp"
parsed = validator.parse_csv_string(csv_string)
print(f"✅ CSV parsing: {len(parsed)} fields extracted")

# TEST 2: RAG System
print("\n2️⃣  TESTING RAG SYSTEM")
print("-" * 60)

print("Skipping rebuild (already built)...")

# Test search
db_path = "/Users/nidhirajani/Desktop/DIC Phase 3/data/external/cve_vector_db"
client = chromadb.PersistentClient(path=str(db_path))
collection = client.get_collection("iot_cves")
model = SentenceTransformer('all-MiniLM-L6-v2')

queries = ["Mirai botnet", "telnet vulnerabilities", "camera exploits"]
for query in queries:
    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )
    print(f"\n✅ Query: '{query}'")
    for meta in results['metadatas'][0]:
        print(f"   - {meta['cve_id']} (CVSS: {meta['cvss_score']})")

# TEST 3: Report Generation
print("\n3️⃣  TESTING REPORT GENERATION")
print("-" * 60)

generator = NetworkSecurityReportGenerator()

# Sample data
network_stats = {
    'total': 76671,
    'malicious': 47706,
    'benign': 28965,
    'threat_level': 'HIGH'
}

malware_distribution = {
    'PortScan': 29814,
    'Okiru': 9882,
    'DDoS': 4494,
    'C&C': 3331
}

top_threats = [
    {'family': 'PortScan', 'count': 29814, 'percentage': 62.5},
    {'family': 'Okiru', 'count': 9882, 'percentage': 20.7},
    {'family': 'DDoS', 'count': 4494, 'percentage': 9.4}
]

cve_mappings = [
    {'cve_id': 'CVE-2016-10401', 'cvss': 9.8, 'description': 'Mirai botnet telnet exploit allowing remote code execution through default credentials'},
    {'cve_id': 'CVE-2017-17215', 'cvss': 8.8, 'description': 'Router remote code execution vulnerability exploited by IoT botnets'}
]

recommendations = [
    "Patch all CVEs with CVSS scores above 7.0 immediately",
    "Block identified C&C server IP addresses at firewall level",
    "Change default passwords on all IoT devices"
]

# Generate PDF
pdf_path = generator.generate_report(
    network_stats, malware_distribution, top_threats,
    cve_mappings, recommendations
)
print(f"✅ PDF generated: {pdf_path}")

# Generate Markdown
md_path = generator.generate_markdown_report(
    network_stats, malware_distribution, top_threats,
    cve_mappings, recommendations
)
print(f"✅ Markdown generated: {md_path}")

print("\n" + "=" * 60)
print("✅ ALL TESTS COMPLETE!")
print("=" * 60)