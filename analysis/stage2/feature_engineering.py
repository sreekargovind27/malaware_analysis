"""
Stage 2: Feature Engineering - PySpark Version
Reimplements the original pandas pipeline faithfully in Spark.

What we do:
- tolerant CSV load across inconsistent IoT-23 headers
- label parsing (label / attack_type / malware_family / attack_subtype)
- numeric cleanup + _was_missing + log1p
- IP-derived features (per orig/resp, plus device_ip)
- custom behavioral features (upload_ratio, packet_rate, suspicious_score, etc.)
- categorical encoding
    - service/history => label index
    - proto/conn_state => one-hot vecs
    - is_S0_state, suspicious_score
- final column list is saved to feature_list.joblib for later steps
- write unified parquet to Config.ENGINEERED_DATA_PATH
"""

import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

from config import Config


# --------------------------------------------------------------------------------
# RAW LOAD (tolerant to messy IoT-23 headers)
# --------------------------------------------------------------------------------
def load_raw_flows(spark):
    """
    Read ALL CSVs from Config.RAW_DIR_ORIGINAL.
    Tolerant to inconsistent headers.
    Normalizes expected columns, enforces types, and adds Source_Folder.
    """
    input_glob = os.path.join(Config.RAW_DIR_ORIGINAL, "*.csv")
    print(f"📂 Reading raw CSVs from {input_glob}")

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("comment", "#")
        .option("mode", "PERMISSIVE")
        .csv(input_glob)
    )

    # keep which capture each row came from
    df = df.withColumn(
        "Source_Folder",
        F.regexp_extract(F.input_file_name(), r"([^/]+)\.csv$", 1)
    )

    # All columns we expect downstream
    expected_cols = [
        "ts", "uid",
        "id.orig_h", "id.orig_p",
        "id.resp_h", "id.resp_p",
        "proto", "service",
        "duration",
        "orig_bytes", "resp_bytes",
        "conn_state",
        "local_orig", "local_resp",
        "missed_bytes",
        "history",
        "orig_pkts", "orig_ip_bytes",
        "resp_pkts", "resp_ip_bytes",
        "label", "detailed-label",
    ]

    # map underscore forms -> dotted forms if needed
    existing_cols = df.columns
    for col_dot in expected_cols:
        alt = col_dot.replace(".", "_").replace("-", "_")
        if col_dot not in existing_cols and alt in existing_cols:
            df = df.withColumnRenamed(alt, col_dot)

    # ensure missing expected cols exist
    for c in expected_cols:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast("string"))

    # helper: safely cast dotted numeric columns
    def safe_cast_numeric(df_in, col_name, target_type):
        """
        Spark hates withColumn('a.b', ...) because it parses `a`.`b`.
        Workaround:
        - create temp col with underscore
        - drop original
        - rename temp back
        Only run this if the column actually exists.
        """
        if col_name not in df_in.columns:
            return df_in

        temp_col = col_name.replace(".", "_") + "__casttmp"

        # Use backticks to escape column names with dots
        df_tmp = df_in.withColumn(temp_col, F.col(f"`{col_name}`").cast(target_type))
        df_tmp = df_tmp.drop(col_name)
        df_tmp = df_tmp.withColumnRenamed(temp_col, col_name)
        return df_tmp

    # cast integer-ish columns to long
    int_like = [
        "id.orig_p", "id.resp_p",
        "orig_bytes", "resp_bytes",
        "orig_pkts", "resp_pkts",
        "orig_ip_bytes", "resp_ip_bytes",
        "missed_bytes",
    ]
    for c in int_like:
        df = safe_cast_numeric(df, c, "long")

    # cast duration, ts to double
    df = safe_cast_numeric(df, "duration", "double")
    df = safe_cast_numeric(df, "ts", "double")

    # local_orig/local_resp -> 0/1 int
    for c in ["local_orig", "local_resp"]:
        if c in df.columns:
            # Cast to string first, then check for "1", "T", "true" (case-insensitive)
            df = df.withColumn(
                c,
                F.when(
                    F.upper(F.col(c).cast("string")).isin("1", "T", "TRUE"),
                    F.lit(1)
                )
                .otherwise(F.lit(0))
                .cast("int")
            )

    return df


# --------------------------------------------------------------------------------
# LABEL PARSING (Stage1-aligned logic, Spark-safe)
# --------------------------------------------------------------------------------
def clean_and_expand_labels(df):
    """
    Produces:
      - label ("Benign"/"Malicious")
      - attack_type (PortScan, C&C, DDoS, Malware, etc.; "Benign" if benign)
      - malware_family (Mirai / Okiru / Torii / Kenjiro / Hajime / etc.)
      - attack_subtype (HeartBeat, FileDownload)
    """
    print("🧠 Cleaning / expanding labels...")

    # Build combined text ONCE using both label and detailed-label
    combined_expr = F.lower(
        F.concat_ws(
            " ",
            F.coalesce(F.col("label"), F.lit("")),
            F.coalesce(F.col("`detailed-label`"), F.lit(""))
        )
    )
    combined_expr = F.regexp_replace(combined_expr, r"\(empty\)", "")
    combined_expr = F.regexp_replace(combined_expr, r"-", " ")
    combined_expr = F.trim(combined_expr)

    # cache it on the df
    df = df.withColumn("combined_text", combined_expr)

    # 1. Binary label
    df = df.withColumn(
        "label",
        F.when(F.col("combined_text").contains("malicious"), "Malicious")
        .when(F.col("combined_text").contains("benign"), "Benign")
        .when(
            (F.col("label").isNotNull()) &
            (F.col("label") != "") &
            (F.col("label") != "-"),
            F.when(F.lower(F.col("label")).contains("malicious"), "Malicious")
            .when(F.lower(F.col("label")).contains("benign"), "Benign")
            .otherwise(F.lit(None).cast("string"))
        )
        .otherwise(F.lit(None).cast("string"))
    )

    # 2. Initial malware_family guess from free text
    df = df.withColumn(
        "malware_family",
        F.when(F.col("combined_text").rlike("mirai"), "Mirai")
        .when(F.col("combined_text").rlike("okiru"), "Okiru")
        .when(F.col("combined_text").rlike("torii"), "Torii")
        .when(F.col("combined_text").rlike("kenjiro"), "Kenjiro")
        .when(F.col("combined_text").rlike("hajime"), "Hajime")
        .when(F.col("combined_text").rlike("gagfyt|gafgyt"), "Gagfyt")
        .when(F.col("combined_text").rlike("muhstik"), "Muhstik")
        .when(F.col("combined_text").rlike("hakai"), "Hakai")
        .when(F.col("combined_text").rlike("ircbot"), "IRCBot")
        .when(F.col("combined_text").rlike("hide.*seek|hideandseek"), "Hide and Seek")
        .when(F.col("combined_text").rlike("trojan"), "Trojan")
        .otherwise(F.lit(None).cast("string"))
    )

    # 3. attack_subtype
    df = df.withColumn(
        "attack_subtype",
        F.when(F.col("combined_text").rlike("heartbeat"), "HeartBeat")
        .when(F.col("combined_text").rlike("filedownload"), "FileDownload")
        .otherwise(F.lit(None).cast("string"))
    )

    # 4. filename-based malware_family override (ground truth per capture)
    print("🗂️  Applying filename-based family labels (ground truth)...")

    family_map = Config.FILENAME_TO_FAMILY_MAP

    @F.udf(T.StringType())
    def get_family_from_filename(source_folder):
        return family_map.get(source_folder, None)

    df = df.withColumn(
        "malware_family_filename",
        get_family_from_filename(F.col("Source_Folder"))
    )

    df = df.withColumn(
        "malware_family",
        F.when(
            F.col("label") == "Malicious",
            F.coalesce(F.col("malware_family_filename"), F.col("malware_family"))
        ).otherwise(F.lit(None).cast("string"))
    )

    # 5. attack_type (fine-grained, Stage1-style)
    # Priority order matters. We try the specific ones first.
    df = df.withColumn(
        "attack_type",
        F.when(F.col("label") == "Benign", "Benign")
        .when(
            F.col("combined_text").contains("portscan") |
            F.col("combined_text").contains("partofahorizontalportscan"),
            "PortScan"
        )
        .when(
            F.col("combined_text").contains("c&c") |
            F.col("combined_text").contains("c2"),
            "C&C"
        )
        .when(
            F.col("combined_text").contains("ddos"),
            "DDoS"
        )
        .when(
            F.col("combined_text").contains("filedownload"),
            "FileDownload"
        )
        .when(
            F.col("combined_text").contains("attack"),
            "Attack"
        )
        # fallback bucket for malicious that didn't match above
        .when(F.col("label") == "Malicious", "Malware")
        .otherwise("Unknown")
    )

    # 6. now safe to drop helper cols
    df = df.drop("detailed-label", "combined_text", "malware_family_filename")

    print("✓ Labels parsed into {label, attack_type, attack_subtype, malware_family}")
    print("✓ Malware families assigned from filenames (ground truth)")
    return df


# --------------------------------------------------------------------------------
# NUMERIC FEATURE CLEANUP
# --------------------------------------------------------------------------------
def engineer_numeric_features(df):
    """
    For every col in Config.BASE_NUMERICAL_FEATURES:
      - create <col>_was_missing (1 if null, else 0)
      - fill null/negative with 0
      - cast to double
      - log1p skewed cols in Config.SKEWED_NUMERICAL_FEATURES
      - if column didn't exist at all, create it with zeros + was_missing=1
    """
    print("🔧 Engineering numeric/base features...")

    for col_name in Config.BASE_NUMERICAL_FEATURES:
        if col_name in df.columns:
            col_ref = f"`{col_name}`"

            # mark missing
            df = df.withColumn(
                f"{col_name}_was_missing",
                F.when(F.col(col_ref).isNull(), 1).otherwise(0).cast("int")
            )

            # clamp negatives/nulls -> 0, cast to double
            df = df.withColumn(
                col_name,
                F.when(F.col(col_ref).isNull(), 0)
                .when(F.col(col_ref) < 0, 0)
                .otherwise(F.col(col_ref).cast("double"))
            )

            # log1p on the CLEANED version
            if col_name in Config.SKEWED_NUMERICAL_FEATURES:
                df = df.withColumn(col_name, F.log1p(F.col(col_name)))

        else:
            # column missing completely
            df = (
                df.withColumn(col_name, F.lit(0.0).cast("double"))
                .withColumn(f"{col_name}_was_missing", F.lit(1).cast("int"))
            )

    return df


# --------------------------------------------------------------------------------
# IP FEATURES
# --------------------------------------------------------------------------------
def extract_ip_features(df):
    print("🌐 Extracting IP features...")

    def first_octet_expr(col):
        return F.split(col, r"\.").getItem(0).cast("int")

    def second_octet_expr(col):
        return F.split(col, r"\.").getItem(1).cast("int")

    # preserve device_ip for downstream aggregation & graph
    if "id.orig_h" in df.columns:
        df = df.withColumn("device_ip", F.col("`id.orig_h`").cast("string"))
    elif "id_orig_h" in df.columns:
        df = df.withColumn("device_ip", F.col("id_orig_h").cast("string"))
    else:
        df = df.withColumn("device_ip", F.lit(None).cast("string"))

    for prefix, ip_col in [("orig", "id.orig_h"), ("resp", "id.resp_h")]:
        if ip_col not in df.columns:
            continue

        col_ref = f"`{ip_col}`"
        clean_ip = F.coalesce(F.col(col_ref).cast("string"), F.lit("0.0.0.0"))

        first_oct = first_octet_expr(clean_ip)
        second_oct = second_octet_expr(clean_ip)

        df = df.withColumn(
            f"{prefix}_is_private",
            (
                    (first_oct == 10) |
                    (first_oct == 192) |
                    ((first_oct == 172) & (second_oct.between(16, 31)))
            ).cast("int")
        )

        df = df.withColumn(
            f"{prefix}_is_broadcast",
            (clean_ip == F.lit("255.255.255.255")).cast("int")
        )

        df = df.withColumn(
            f"{prefix}_is_multicast",
            ((first_oct >= 224) & (first_oct <= 239)).cast("int")
        )

        df = df.withColumn(
            f"{prefix}_is_localhost",
            (first_oct == 127).cast("int")
        )

        df = df.withColumn(
            f"{prefix}_ip_first_octet",
            first_oct
        )

    # DO NOT drop id.resp_h / id.orig_h
    # graph builder consumes them
    return df


# --------------------------------------------------------------------------------
# CUSTOM / BEHAVIORAL FEATURES
# --------------------------------------------------------------------------------
def engineer_custom_features(df):
    print("🛠 Creating traffic behavior features...")

    total_bytes = (F.col("orig_bytes") + F.col("resp_bytes")).cast("double")
    total_pkts = (F.col("orig_pkts") + F.col("resp_pkts")).cast("double")

    # upload_ratio
    df = df.withColumn(
        "upload_ratio",
        (F.col("orig_bytes").cast("double") / (total_bytes + F.lit(1e-9)))
    )

    # bytes_per_packet
    df = df.withColumn(
        "bytes_per_packet",
        (total_bytes / (total_pkts + F.lit(1e-9)))
    )

    # packet_rate
    safe_duration = F.expm1(F.col("duration"))
    safe_duration = F.when(safe_duration < 0.001, 0.001).otherwise(safe_duration)

    df = df.withColumn(
        "packet_rate",
        F.when(
            safe_duration > 0,
            (total_pkts / safe_duration).cast("double")
        ).otherwise(F.lit(0.0))
    )

    # clip packet_rate to 10000
    df = df.withColumn(
        "packet_rate",
        F.when(F.col("packet_rate") > 10000, 10000.0)
        .otherwise(F.col("packet_rate"))
    )

    # port-based flags
    df = df.withColumn("is_port_23", (F.col("`id.resp_p`") == 23).cast("int"))
    df = df.withColumn("is_port_22", (F.col("`id.resp_p`") == 22).cast("int"))

    # placeholders from pandas
    df = df.withColumn("is_telnet", F.lit(0).cast("int"))
    df = df.withColumn("is_unknown_service", F.lit(0).cast("int"))

    return df


# --------------------------------------------------------------------------------
# CATEGORICAL ENCODING
# --------------------------------------------------------------------------------
def engineer_categorical_features(df):
    print("🏅 Encoding categorical features...")

    def normalize(colname):
        return F.when(
            F.col(colname).isNull() |
            (F.col(colname) == "") |
            (F.col(colname) == "-") |
            (F.col(colname) == "nan") |
            (F.col(colname) == "None"),
            F.lit("unknown")
        ).otherwise(F.col(colname).cast("string"))

    for c in ["proto", "conn_state", "service", "history"]:
        if c in df.columns:
            df = df.withColumn(c, normalize(c))
        else:
            df = df.withColumn(c, F.lit("unknown"))

    stages = []

    # service/history -> index only
    for c in ["service", "history"]:
        idx_col = f"{c}_idx"
        stages.append(
            StringIndexer(
                inputCol=c,
                outputCol=idx_col,
                handleInvalid="keep"
            )
        )

    # proto/conn_state -> OHE
    for c in ["proto", "conn_state"]:
        idx_col = f"{c}_idx"
        vec_col = f"{c}_vec"
        stages.append(
            StringIndexer(
                inputCol=c,
                outputCol=idx_col,
                handleInvalid="keep"
            )
        )
        stages.append(
            OneHotEncoder(
                inputCol=idx_col,
                outputCol=vec_col
            )
        )

    pipe = Pipeline(stages=stages)
    model = pipe.fit(df)
    df = model.transform(df)

    # is_S0_state
    df = df.withColumn("is_S0_state", (F.col("conn_state") == "S0").cast("int"))

    # suspicious_score = is_port_23*40 + is_S0_state*30 + is_port_22*25
    df = df.withColumn(
        "suspicious_score",
        F.col("is_port_23") * F.lit(40) +
        F.col("is_S0_state") * F.lit(30) +
        F.col("is_port_22") * F.lit(25)
    )

    # DO NOT drop id.resp_p; graph builder needs it
    return df


# --------------------------------------------------------------------------------
# FEATURE LIST
# --------------------------------------------------------------------------------
def build_and_save_feature_list(df):
    print("📎 Building feature list...")

    target_cols = {
        Config.TARGET_COL,  # "label"
        Config.DETAILED_TARGET_COL,  # "attack_type"
        Config.FAMILY_TARGET_COL,  # "malware_family"
        "attack_subtype"
    }

    id_like_cols = {
        "device_ip",
        "uid",
        "Source_Folder",
        "ts"
    }

    feature_cols = [
        c for c, t in df.dtypes
        if c not in target_cols and c not in id_like_cols
    ]

    joblib.dump(feature_cols, Config.FEATURE_LIST_PATH)
    print(f"✓ Saved {len(feature_cols)} features to {Config.FEATURE_LIST_PATH}")


# --------------------------------------------------------------------------------
# MAIN ENTRYPOINT
# --------------------------------------------------------------------------------
def build_engineered_dataset_spark(spark):
    print("\n" + "=" * 70)
    print("🚀 PYSPARK FEATURE ENGINEERING PIPELINE")
    print("=" * 70)
    print(f"Mode: {'TEST' if Config.TEST_MODE else 'PRODUCTION'}")
    print(f"Input dir: {Config.RAW_DIR_ORIGINAL}")
    print(f"Output parquet: {Config.ENGINEERED_DATA_PATH}")
    print("=" * 70)

    # ensure output dirs exist
    Config.ensure_output_dirs()

    # 1. Load raw
    df = load_raw_flows(spark)
    raw_count = df.count()
    print(f"✓ Loaded {raw_count:,} raw rows")
    if raw_count == 0:
        print("❌ No data found. Aborting feature engineering.")
        return

    # 2. Parse/align labels, families, attack types
    df = clean_and_expand_labels(df)

    # 3. Drop rows where label is NULL (unlabeled junk)
    print("🧹 Filtering out rows with NULL labels...")
    rows_before = df.count()
    df = df.filter(F.col("label").isNotNull())
    rows_after = df.count()
    rows_dropped = rows_before - rows_after
    print(f"✓ Dropped {rows_dropped:,} rows with NULL labels (kept {rows_after:,})")

    # 4. Numeric cleanup
    df = engineer_numeric_features(df)

    # 5. IP features (+ device_ip)
    df = extract_ip_features(df)

    # 6. Behavioral/custom features
    df = engineer_custom_features(df)

    # 7. Categorical encoding
    df = engineer_categorical_features(df)

    # Cache for safety
    df = df.cache()
    final_rows = df.count()
    print(f"✓ Engineered rows: {final_rows:,}")

    # 8. Save feature list
    build_and_save_feature_list(df)

    # 9. Write final parquet
    print("\n💾 Saving engineered dataset...")
    (
        df.write
        .mode("overwrite")
        .parquet(Config.ENGINEERED_DATA_PATH, compression="snappy")
    )
    print(f"✓ Saved to {Config.ENGINEERED_DATA_PATH}")

    print("\n✅ FEATURE ENGINEERING COMPLETE!")
    print("=" * 70)

    return df


if __name__ == "__main__":
    spark = Config.get_spark_session("Stage2-FeatureEngineering")
    build_engineered_dataset_spark(spark)
    spark.stop()
