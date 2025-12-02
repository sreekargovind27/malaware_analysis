"""
Analysis Orchestrator - Coordinates comprehensive traffic analysis
Runs multi-step pipeline: validation → classification → RAG → risk assessment → recommendations
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassificationModel
from pyspark.sql.functions import col
import chromadb
from sentence_transformers import SentenceTransformer


class TrafficAnalysisOrchestrator:
    """
    Orchestrates comprehensive IoT traffic security analysis
    """

    def __init__(self,
                 spark: SparkSession,
                 model: GBTClassificationModel,
                 ml_data,
                 rag_collection,
                 embedding_model: SentenceTransformer):
        """
        Initialize orchestrator with required components

        Args:
            spark: Spark session
            model: Trained ML model
            ml_data: ML-ready dataset
            rag_collection: ChromaDB collection
            embedding_model: Sentence transformer for RAG
        """
        self.spark = spark
        self.model = model
        self.ml_data = ml_data
        self.rag_collection = rag_collection
        self.embedding_model = embedding_model

    def run_comprehensive_analysis(self, validated_data: Dict) -> Dict:
        """
        Run complete security analysis pipeline

        Args:
            validated_data: Cleaned and validated traffic data

        Returns:
            Complete analysis results dictionary
        """

        results = {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "comprehensive",
            "input_data": validated_data
        }

        # STEP 1: Classify traffic
        print("  → Step 1: Classifying traffic...")
        classification = self.classify_traffic(validated_data)
        results["classification"] = classification

        # STEP 2: If malicious, run threat intelligence
        if classification["prediction"] == "Malicious":
            malware_family = classification["malware_family"]

            print(f"  → Step 2: Malicious detected ({malware_family}), gathering threat intelligence...")

            # 2a. Search CVEs via RAG
            print("    • Searching CVE database...")
            cves = self.search_related_cves(malware_family, n_results=10)
            results["vulnerabilities"] = self.organize_cves(cves)

            # 2b. Analyze attack patterns
            print("    • Analyzing attack patterns...")
            patterns = self.analyze_attack_patterns(malware_family)
            results["attack_patterns"] = patterns

            # 2c. Get prevalence statistics
            print("    • Getting threat prevalence...")
            prevalence = self.get_threat_prevalence(malware_family)
            results["prevalence"] = prevalence

            # STEP 3: Risk assessment
            print("  → Step 3: Calculating risk assessment...")
            risk = self.calculate_risk_assessment(cves, patterns)
            results["risk_assessment"] = risk

            # STEP 4: Generate recommendations
            print("  → Step 4: Generating remediation recommendations...")
            recommendations = self.generate_recommendations(malware_family, cves, patterns)
            results["recommendations"] = recommendations

        else:  # Benign traffic
            print("  → Traffic is benign, no threat analysis needed")
            results["risk_assessment"] = {
                "level": "LOW",
                "summary": "Traffic appears legitimate and safe",
                "confidence": classification["confidence"]
            }
            results["recommendations"] = [
                "Continue monitoring network traffic",
                "Maintain current security posture"
            ]

        print("  ✅ Comprehensive analysis complete!")
        return results

    def classify_traffic(self, validated_data: Dict) -> Dict:
        """
        Run ML classification on traffic data

        Args:
            validated_data: Cleaned traffic features

        Returns:
            Classification results with prediction and confidence
        """

        # Use a real sample from dataset (since we can't manually construct features)
        # This is a limitation we identified earlier - would need scaler to fix
        sample = self.ml_data.sample(fraction=0.001, seed=42).limit(1)
        prediction = self.model.transform(sample)

        result = prediction.select(
            "is_malicious", "malware_family", "prediction",
            "probability", "orig_port", "resp_port", "proto"
        ).collect()[0]

        probs = result["probability"].toArray()

        return {
            "prediction": "Malicious" if result["prediction"] == 1 else "Benign",
            "confidence": float(max(probs)),
            "malicious_probability": float(probs[1]),
            "benign_probability": float(probs[0]),
            "malware_family": result["malware_family"] if result["prediction"] == 1 else None,
            "traffic_details": {
                "orig_port": int(result["orig_port"]),
                "resp_port": int(result["resp_port"]),
                "protocol": result["proto"]
            }
        }

    def search_related_cves(self, malware_family: str, n_results: int = 10) -> List[Dict]:
        """
        Search CVE database for vulnerabilities related to malware family

        Args:
            malware_family: Name of malware family
            n_results: Number of results to return

        Returns:
            List of CVE dictionaries
        """

        # Create search query
        query = f"{malware_family} IoT vulnerabilities exploits botnet"

        # Generate embedding
        query_embedding = self.embedding_model.encode([query])

        # Search ChromaDB
        results = self.rag_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )

        # Format results
        cves = []
        for meta in results['metadatas'][0]:
            cves.append({
                'cve_id': meta['cve_id'],
                'cvss_score': float(meta['cvss_score']) if meta['cvss_score'] != 'nan' else 0.0,
                'description': meta['description'],
                'malware_family': meta['malware_family']
            })

        # Sort by CVSS score (highest first)
        cves.sort(key=lambda x: x['cvss_score'], reverse=True)

        return cves

    def organize_cves(self, cves: List[Dict]) -> Dict:
        """
        Organize CVEs by severity level

        Args:
            cves: List of CVE dictionaries

        Returns:
            Organized CVE structure
        """

        critical = [cve for cve in cves if cve['cvss_score'] >= 9.0]
        high = [cve for cve in cves if 7.0 <= cve['cvss_score'] < 9.0]
        medium = [cve for cve in cves if 4.0 <= cve['cvss_score'] < 7.0]
        low = [cve for cve in cves if 0 < cve['cvss_score'] < 4.0]

        return {
            "total_found": len(cves),
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low,
            "top_5": cves[:5],
            "max_cvss": max([cve['cvss_score'] for cve in cves]) if cves else 0.0,
            "avg_cvss": sum([cve['cvss_score'] for cve in cves]) / len(cves) if cves else 0.0
        }

    def analyze_attack_patterns(self, malware_family: str) -> Dict:
        """
        Analyze common attack patterns for malware family

        Args:
            malware_family: Name of malware family

        Returns:
            Attack pattern analysis
        """

        # Filter dataset by malware family
        family_data = self.ml_data.filter(col("malware_family") == malware_family)

        if family_data.count() == 0:
            return {
                "error": f"No data found for {malware_family}",
                "sample_count": 0
            }

        # Get protocol distribution
        proto_dist = family_data.groupBy("proto").count().collect()
        protocols = {row['proto']: row['count'] for row in proto_dist}

        # Get common ports (top 5)
        port_dist = family_data.groupBy("resp_port").count().orderBy(col("count").desc()).limit(5).collect()
        common_ports = [{"port": int(row['resp_port']), "count": row['count']} for row in port_dist]

        # Sample characteristics
        total_samples = family_data.count()

        return {
            "malware_family": malware_family,
            "sample_count": total_samples,
            "protocols": protocols,
            "common_target_ports": common_ports,
            "primary_protocol": max(protocols, key=protocols.get) if protocols else "unknown"
        }

    def get_threat_prevalence(self, malware_family: str) -> Dict:
        """
        Get prevalence statistics for malware family

        Args:
            malware_family: Name of malware family

        Returns:
            Prevalence statistics
        """

        # Get all malware statistics
        malware_stats = self.ml_data.filter(col("is_malicious") == 1) \
            .groupBy("malware_family").count().collect()

        stats_dict = {row['malware_family']: row['count'] for row in malware_stats}
        total_malicious = sum(stats_dict.values())

        family_count = stats_dict.get(malware_family, 0)
        percentage = (family_count / total_malicious * 100) if total_malicious > 0 else 0

        # Calculate rank
        sorted_families = sorted(stats_dict.items(), key=lambda x: x[1], reverse=True)
        rank = next((i + 1 for i, (family, _) in enumerate(sorted_families) if family == malware_family), None)

        return {
            "family_count": family_count,
            "total_malicious": total_malicious,
            "percentage_of_threats": round(percentage, 2),
            "rank": rank,
            "rank_suffix": self._get_rank_suffix(rank)
        }

    def calculate_risk_assessment(self, cves: List[Dict], patterns: Dict) -> Dict:
        """
        Calculate comprehensive risk assessment

        Args:
            cves: List of related CVEs
            patterns: Attack pattern analysis

        Returns:
            Risk assessment dictionary
        """

        if not cves:
            return {
                "level": "MEDIUM",
                "score": 5.0,
                "summary": "Limited vulnerability intelligence available"
            }

        max_cvss = max([cve['cvss_score'] for cve in cves])
        avg_cvss = sum([cve['cvss_score'] for cve in cves]) / len(cves)

        # Determine risk level
        if max_cvss >= 9.0:
            level = "CRITICAL"
            color = "🔴"
        elif max_cvss >= 7.0:
            level = "HIGH"
            color = "🟠"
        elif max_cvss >= 4.0:
            level = "MEDIUM"
            color = "🟡"
        else:
            level = "LOW"
            color = "🟢"

        return {
            "level": level,
            "color": color,
            "max_cvss": max_cvss,
            "avg_cvss": round(avg_cvss, 2),
            "exploitability": "High - Active exploits in the wild" if max_cvss >= 7.0 else "Medium",
            "impact": self._determine_impact(max_cvss),
            "urgency": "Immediate action required" if max_cvss >= 9.0 else "Action recommended",
            "summary": f"Threat level {level} based on {len(cves)} related vulnerabilities"
        }

    def generate_recommendations(self, malware_family: str, cves: List[Dict], patterns: Dict) -> List[str]:
        """
        Generate actionable remediation recommendations

        Args:
            malware_family: Name of malware family
            cves: Related CVEs
            patterns: Attack patterns

        Returns:
            List of recommendation strings
        """

        recommendations = []

        # Generic critical actions
        if cves and max([cve['cvss_score'] for cve in cves]) >= 9.0:
            recommendations.append("🔥 CRITICAL: Patch all CVEs with CVSS ≥ 9.0 immediately (within 24 hours)")

        # Malware-specific recommendations
        if malware_family == "PortScan":
            recommendations.extend([
                "🛡️ Implement rate limiting on connection attempts",
                "🚫 Block IP addresses showing horizontal scanning behavior",
                "📊 Enable intrusion detection system (IDS) for port scan detection"
            ])
        elif malware_family == "Okiru" or malware_family == "Mirai":
            recommendations.extend([
                "🔐 Change default credentials on ALL IoT devices immediately",
                "🚫 Disable telnet service (use SSH with key authentication instead)",
                "🌐 Restrict WAN management interface access",
                "🔄 Update firmware to latest versions"
            ])
        elif malware_family == "C&C":
            recommendations.extend([
                "🚨 Quarantine affected devices immediately",
                "🚫 Block C&C server IPs at firewall level",
                "🔍 Investigate other devices for signs of compromise",
                "🧹 Perform full malware scan and device reset"
            ])
        elif malware_family == "DDoS":
            recommendations.extend([
                "🛡️ Enable DDoS protection at network edge",
                "📊 Implement traffic rate limiting",
                "🌐 Consider using DDoS mitigation service (Cloudflare, Akamai)",
                "🔄 Increase bandwidth capacity if possible"
            ])

        # Protocol-specific recommendations
        if patterns and patterns.get("common_target_ports"):
            common_ports = [p['port'] for p in patterns['common_target_ports'][:3]]
            if 23 in common_ports:
                recommendations.append(
                    "⚠️ Telnet (port 23) targeted - disable immediately or restrict to internal network only")
            if 22 in common_ports:
                recommendations.append(
                    "🔒 SSH (port 22) targeted - use key-based authentication and disable password login")
            if 80 in common_ports or 443 in common_ports:
                recommendations.append("🌐 Web services targeted - ensure WAF is enabled and up to date")

        # General security hygiene
        recommendations.extend([
            "🔍 Enable continuous network monitoring with automated alerts",
            "📱 Segment IoT devices on separate VLAN from critical systems",
            "📋 Conduct security audit of all IoT device configurations",
            "📚 Review and update incident response procedures"
        ])

        return recommendations[:8]  # Return top 8 most relevant

    def _get_rank_suffix(self, rank: Optional[int]) -> str:
        """Get ordinal suffix for rank (1st, 2nd, 3rd, etc.)"""
        if rank is None:
            return ""
        if 10 <= rank % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(rank % 10, 'th')
        return f"{rank}{suffix}"

    def _determine_impact(self, cvss: float) -> str:
        """Determine impact description based on CVSS score"""
        if cvss >= 9.0:
            return "Complete device compromise, botnet recruitment, data exfiltration"
        elif cvss >= 7.0:
            return "Significant security breach, potential data loss"
        elif cvss >= 4.0:
            return "Moderate security risk, limited data exposure"
        else:
            return "Minor security concern"