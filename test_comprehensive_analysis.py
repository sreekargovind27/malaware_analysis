"""
Test comprehensive analysis system
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "src" / "mcp"))

print("=" * 70)
print("TESTING COMPREHENSIVE ANALYSIS SYSTEM")
print("=" * 70)

# Test 1: Can we import the new modules?
print("\n1️⃣  Testing imports...")
try:
    from analysis_orchestrator import TrafficAnalysisOrchestrator

    print("   ✅ TrafficAnalysisOrchestrator imported")
except Exception as e:
    print(f"   ❌ Error importing orchestrator: {e}")
    exit(1)

try:
    from response_formatter import format_comprehensive_response, format_quick_summary

    print("   ✅ Response formatters imported")
except Exception as e:
    print(f"   ❌ Error importing formatters: {e}")
    exit(1)

# Test 2: Can we initialize Spark and components?
print("\n2️⃣  Initializing Spark environment...")
try:
    from pyspark.sql import SparkSession
    from pyspark.ml.classification import GBTClassificationModel
    import chromadb
    from sentence_transformers import SentenceTransformer

    spark = SparkSession.builder \
        .appName("Test-Comprehensive") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    print("   ✅ Spark session created")

    # Load model
    model_path = "/Users/nidhirajani/Desktop/DIC Phase 3/models/lightgbm_classifier"
    model = GBTClassificationModel.load(model_path)
    print("   ✅ Model loaded")

    # Load data
    data_path = "/Users/nidhirajani/Desktop/DIC Phase 3/data/processed/ml_ready_data.parquet"
    ml_data = spark.read.parquet(data_path)
    print("   ✅ Data loaded")

    # Load RAG
    db_path = "/Users/nidhirajani/Desktop/DIC Phase 3/data/external/cve_vector_db"
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection("iot_cves")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("   ✅ RAG system loaded")

except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# Test 3: Initialize orchestrator
print("\n3️⃣  Initializing orchestrator...")
try:
    orchestrator = TrafficAnalysisOrchestrator(
        spark=spark,
        model=model,
        ml_data=ml_data,
        rag_collection=collection,
        embedding_model=embedding_model
    )
    print("   ✅ Orchestrator initialized")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# Test 4: Run comprehensive analysis
print("\n4️⃣  Running comprehensive analysis...")
try:
    test_data = {
        'duration': 0.5,
        'orig_bytes': 100,
        'resp_bytes': 200,
        'orig_pkts': 5,
        'resp_pkts': 10,
        'orig_port': 80,
        'resp_port': 443,
        'proto': 'tcp'
    }

    results = orchestrator.run_comprehensive_analysis(test_data)
    print("   ✅ Analysis completed")
    print(f"   Prediction: {results['classification']['prediction']}")
    print(f"   Confidence: {results['classification']['confidence']:.1%}")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Test 5: Format response
print("\n5️⃣  Formatting response...")
try:
    formatted = format_comprehensive_response(results)
    print("   ✅ Response formatted")
    print(f"   Length: {len(formatted)} characters")

    # Show first 500 chars
    print("\n   Preview:")
    print("   " + "-" * 60)
    print("   " + formatted[:500].replace("\n", "\n   "))
    print("   " + "-" * 60)

except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED - SYSTEM READY!")
print("=" * 70)

spark.stop()