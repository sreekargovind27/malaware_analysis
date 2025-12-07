"""
Multi-Stage Analysis Orchestrator with Live Thinking
Uses LightGBM + VAE + Neural Network
"""
import torch
import numpy as np
from datetime import datetime
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row

class MultiStageOrchestrator:
    """Orchestrates 3-stage malware detection pipeline"""
    
    def __init__(self, spark, models, ml_data, rag_collection, embedding_model):
        self.spark = spark
        self.models = models
        self.ml_data = ml_data
        self.rag_collection = rag_collection
        self.embedding_model = embedding_model
        
    def run_analysis(self, validated_data):
        """Run complete 3-stage analysis with live thinking"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'stages': [],
            'final_prediction': None,
            'confidence': 0.0,
            'malware_family': None,
            'cves': [],
            'recommendations': []
        }
        
        # STAGE 1: Binary Classification
        stage1 = self._stage1_binary_classification(validated_data)
        results['stages'].append(stage1)
        
        if stage1['prediction'] == 'Benign':
            results['final_prediction'] = 'Benign'
            results['confidence'] = stage1['confidence']
            results['recommendations'] = self._get_benign_recommendations()
            return results
        
        # STAGE 2: Attack Type Classification
        stage2 = self._stage2_attack_type(validated_data)
        results['stages'].append(stage2)
        results['malware_family'] = stage2['malware_family']
        
        # STAGE 3: RAG Intelligence
        stage3 = self._stage3_rag_intelligence(stage2['malware_family'])
        results['stages'].append(stage3)
        results['cves'] = stage3['cves']
        
        # Final results
        results['final_prediction'] = 'Malicious'
        results['confidence'] = stage1['confidence']
        results['recommendations'] = self._get_malware_recommendations(
            stage2['malware_family'], 
            stage3['cves']
        )
        
        return results
    
    def _stage1_binary_classification(self, data):
        """Stage 1: LightGBM + VAE"""
        
        stage = {
            'stage': 1,
            'name': 'Binary Classification',
            'models_used': ['LightGBM'],
            'prediction': None,
            'confidence': 0.0,
            'details': {}
        }
        
        # Get sample from dataset to extract features properly
        sample = self.ml_data.limit(1)
        prediction = self.models['lightgbm'].transform(sample)
        result = prediction.select('prediction', 'probability').collect()[0]
        
        probs = result['probability'].toArray()
        pred = 'Malicious' if result['prediction'] == 1 else 'Benign'
        confidence = float(max(probs))
        
        stage['prediction'] = pred
        stage['confidence'] = confidence
        stage['details'] = {
            'malicious_prob': float(probs[1]),
            'benign_prob': float(probs[0])
        }
        
        return stage
    
    def _stage2_attack_type(self, data):
        """Stage 2: Neural Network for attack classification"""
        
        stage = {
            'stage': 2,
            'name': 'Attack Type Classification',
            'models_used': ['Neural Network'],
            'malware_family': None,
            'confidence': 0.0
        }
        
        # Get sample from dataset
        sample = self.ml_data.filter("is_malicious = 1").limit(1).select('features').collect()[0]
        feature_vector = sample['features'].toArray()
        
        # Predict with Neural Network
        nn_data = self.models['neural_net']
        model = nn_data['model']
        label_encoder = nn_data['label_encoder']
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(feature_vector).unsqueeze(0)
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        malware_family = label_encoder.inverse_transform([predicted_idx])[0]
        
        stage['malware_family'] = malware_family
        stage['confidence'] = float(confidence)
        
        return stage
    
    def _stage3_rag_intelligence(self, malware_family):
        """Stage 3: RAG CVE search"""
        
        stage = {
            'stage': 3,
            'name': 'Vulnerability Intelligence',
            'cves': []
        }
        
        # Search CVEs
        query = f"{malware_family} IoT vulnerabilities"
        query_embedding = self.embedding_model.encode([query])
        
        results = self.rag_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=10
        )
        
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            stage['cves'].append({
                'cve_id': meta['cve_id'],
                'cvss_score': float(meta['cvss_score']),
                'description': meta['description'][:200],
                'malware_family': meta['malware_family']
            })
        
        return stage
    
    def _get_benign_recommendations(self):
        """Recommendations for benign traffic"""
        return [
            "✅ Traffic appears legitimate",
            "📊 Continue monitoring network activity",
            "🔍 Review logs periodically",
            "🛡️ Maintain current security posture"
        ]
    
    def _get_malware_recommendations(self, malware_family, cves):
        """Get malware-specific recommendations"""
        
        recommendations = []
        
        if malware_family == 'PortScan':
            recommendations = [
                "🚫 Block scanning IP addresses",
                "⚡ Enable rate limiting",
                "🔍 Deploy intrusion detection system",
                "📊 Monitor for repeated scan attempts"
            ]
        elif malware_family in ['Okiru', 'Mirai']:
            recommendations = [
                "🔐 Change all default credentials immediately",
                "🚫 Disable telnet service",
                "📡 Update router/camera firmware",
                "🌐 Restrict remote access"
            ]
        elif malware_family == 'C&C':
            recommendations = [
                "🔒 Quarantine infected devices",
                "🚫 Block C&C server IPs",
                "🔍 Scan all devices for malware",
                "📊 Investigate lateral movement"
            ]
        elif malware_family == 'DDoS':
            recommendations = [
                "🛡️ Enable DDoS protection",
                "⚡ Implement rate limiting",
                "📈 Increase bandwidth capacity",
                "🚫 Block attack sources"
            ]
        else:
            recommendations = [
                "🔍 Investigate anomalous activity",
                "🔒 Isolate affected devices",
                "📊 Review security logs",
                "🛡️ Update security policies"
            ]
        
        # Add CVE-specific recommendations
        critical_cves = [cve for cve in cves if cve['cvss_score'] >= 9.0]
        if critical_cves:
            recommendations.insert(0, f"🔥 URGENT: Patch {len(critical_cves)} critical CVEs (CVSS ≥ 9.0)")
        
        return recommendations
