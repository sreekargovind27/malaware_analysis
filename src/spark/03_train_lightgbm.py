"""
Train LightGBM model for IoT malware classification
Binary classification: Benign (0) vs Malicious (1)
"""

from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import os
import pickle


def create_spark_session():
    """Initialize Spark session"""
    spark = SparkSession.builder \
        .appName("IoT23-LightGBM-Training") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    print(f"✓ Spark session created: {spark.version}")
    return spark


def load_ml_data(spark, data_path):
    """Load ML-ready feature data"""

    ml_data_path = os.path.join(data_path, "ml_ready_data.parquet")

    df = spark.read.parquet(ml_data_path)
    print(f"✓ Loaded {df.count()} rows of ML-ready data")

    # Show class distribution
    print("\nClass distribution:")
    df.groupBy("is_malicious").count().show()

    return df


def prepare_training_data(df):
    """Split data into train/test sets"""

    print("\nSplitting data into train (80%) and test (20%)...")

    # Rename is_malicious to label for Spark ML
    df = df.withColumn("label", col("is_malicious").cast("double"))

    # Split data
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    print(f"✓ Training set: {train_df.count()} rows")
    print(f"✓ Test set: {test_df.count()} rows")

    return train_df, test_df


def train_model(train_df):
    """Train Gradient Boosted Trees (LightGBM alternative in Spark)"""

    print("\nTraining Gradient Boosted Trees classifier...")
    print("(Using Spark's GBT as LightGBM alternative)")

    # Configure GBT classifier
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=100,  # Number of trees
        maxDepth=5,  # Tree depth
        stepSize=0.1,  # Learning rate
        seed=42
    )

    # Train model
    print("Training in progress...")
    model = gbt.fit(train_df)

    print(f"✓ Model trained with {model.getNumTrees} trees")

    return model


def evaluate_model(model, test_df):
    """Evaluate model performance"""

    print("\nEvaluating model on test set...")

    # Make predictions
    predictions = model.transform(test_df)

    # Binary classification metrics
    binary_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    auc = binary_evaluator.evaluate(predictions)
    print(f"✓ AUC-ROC: {auc:.4f}")

    # Multiclass metrics for accuracy, precision, recall
    mc_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction"
    )

    accuracy = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "accuracy"})
    precision = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "weightedPrecision"})
    recall = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "weightedRecall"})
    f1 = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "f1"})

    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ Precision: {precision:.4f}")
    print(f"✓ Recall: {recall:.4f}")
    print(f"✓ F1-Score: {f1:.4f}")

    # Show confusion matrix
    print("\nConfusion Matrix:")
    predictions.groupBy("label", "prediction").count().show()

    # Show sample predictions
    print("\nSample predictions:")
    predictions.select("label", "prediction", "probability", "malware_family").show(10, truncate=False)

    return predictions, {
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def save_model(model, metrics, output_dir):
    """Save trained model and metrics"""

    print("\nSaving model...")

    # Create models directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save Spark model
    model_path = os.path.join(output_dir, "lightgbm_classifier")
    model.write().overwrite().save(model_path)
    print(f"✓ Spark model saved: {model_path}")

    # Save metrics
    metrics_path = os.path.join(output_dir, "lightgbm_metrics.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"✓ Metrics saved: {metrics_path}")

    # Save feature importance
    if hasattr(model, 'featureImportances'):
        importance = model.featureImportances
        importance_path = os.path.join(output_dir, "feature_importance.txt")
        with open(importance_path, 'w') as f:
            f.write("Feature Importances:\n")
            f.write(str(importance))
        print(f"✓ Feature importance saved: {importance_path}")


def main():
    """Main execution flow"""

    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    processed_data_path = os.path.join(project_root, "data", "processed")
    models_path = os.path.join(project_root, "models")

    print("=" * 60)
    print("LightGBM Model Training - IoT Malware Classification")
    print("=" * 60)

    # Step 1: Create Spark session
    spark = create_spark_session()

    # Step 2: Load ML-ready data
    df = load_ml_data(spark, processed_data_path)

    # Step 3: Prepare train/test split
    train_df, test_df = prepare_training_data(df)

    # Step 4: Train model
    model = train_model(train_df)

    # Step 5: Evaluate model
    predictions, metrics = evaluate_model(model, test_df)

    # Step 6: Save model
    save_model(model, metrics, models_path)

    print("\n" + "=" * 60)
    print("✓ Model training complete!")
    print(f"✓ Model saved to: {models_path}")
    print(f"✓ AUC-ROC: {metrics['auc']:.4f}")
    print(f"✓ Accuracy: {metrics['accuracy']:.4f}")
    print(f"✓ F1-Score: {metrics['f1']:.4f}")
    print("=" * 60)

    # Stop Spark
    spark.stop()


if __name__ == "__main__":
    main()