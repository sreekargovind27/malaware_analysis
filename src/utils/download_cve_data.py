"""
Download and process CVE data for IoT vulnerabilities
"""
import requests
import json
import pandas as pd
import os
import time


def download_cve_iot_data():
    """Download IoT-related CVEs using NVD API 2.0"""

    output_dir = "data/external"
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading IoT CVE data from NVD API...")

    # IoT-related keywords
    iot_keywords = ['mirai', 'camera', 'router', 'iot', 'telnet']

    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

    all_cves = []

    for keyword in iot_keywords:
        print(f"\nSearching for: {keyword}")

        params = {
            'keywordSearch': keyword,
            'resultsPerPage': 100
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()

                if 'vulnerabilities' in data:
                    vulns = data['vulnerabilities']
                    print(f"  Found {len(vulns)} CVEs")
                    all_cves.extend(vulns)
                else:
                    print(f"  No vulnerabilities found")

            elif response.status_code == 403:
                print(f"  Rate limited. Waiting 10 seconds...")
                time.sleep(10)

            else:
                print(f"  Error: Status {response.status_code}")

        except Exception as e:
            print(f"  Error: {e}")

        # Be nice to NVD API
        time.sleep(6)  # Required: max 5 requests per 30 seconds

    # Process CVEs
    cve_records = []

    for item in all_cves:
        cve = item.get('cve', {})
        cve_id = cve.get('id', 'N/A')

        # Get description
        descriptions = cve.get('descriptions', [])
        description = descriptions[0].get('value', '') if descriptions else ''

        # Get published date
        published = cve.get('published', 'N/A')

        # Get CVSS score if available
        metrics = cve.get('metrics', {})
        cvss_score = 'N/A'

        if 'cvssMetricV31' in metrics and len(metrics['cvssMetricV31']) > 0:
            cvss_score = metrics['cvssMetricV31'][0].get('cvssData', {}).get('baseScore', 'N/A')

        cve_records.append({
            'cve_id': cve_id,
            'description': description,
            'published_date': published,
            'cvss_score': cvss_score
        })

    # Remove duplicates
    df = pd.DataFrame(cve_records)
    df = df.drop_duplicates(subset=['cve_id'])

    # Map CVEs to malware families
    df['malware_family'] = 'Unknown'
    df.loc[df['description'].str.contains('Mirai|mirai', case=False, na=False), 'malware_family'] = 'Mirai'
    df.loc[df['description'].str.contains('telnet|default password', case=False, na=False), 'malware_family'] = 'Mirai'

    # Save as CSV
    output_file = os.path.join(output_dir, 'iot_cves.csv')
    df.to_csv(output_file, index=False)

    print(f"\n{'=' * 60}")
    print(f"✓ Downloaded {len(df)} unique CVEs")
    print(f"✓ Saved to: {output_file}")
    print(f"{'=' * 60}")

    # Show sample
    print("\nSample CVEs:")
    print(df.head())

    return df


if __name__ == "__main__":
    download_cve_iot_data()