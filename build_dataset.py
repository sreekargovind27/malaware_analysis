"""
OPTIMIZED: Parallel processing + faster I/O + better chunking
Expected speedup: 3-5x faster (20-40 min → 5-10 min)
"""
import glob
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import dask.dataframe as dd
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config

# ==================== OPTIMIZATION 1: LARGER CHUNK SIZE ====================
CHUNK_SIZE = 1500000  # 1.5 mil

# ==================== OPTIMIZATION 2: USE ALL CPU CORES ====================
MAX_WORKERS = min(8, cpu_count() - 1)


# ==================== MOVE TO MODULE LEVEL FOR PICKLING ====================
def scan_single_file(file_path):
    """Scan one file for categorical values - MUST BE AT MODULE LEVEL"""
    proto_values = set()
    conn_state_values = set()

    try:
        df_sample = pd.read_csv(
            file_path,
            usecols=['proto', 'conn_state'],
            low_memory=False,
            on_bad_lines='skip'
        )

        for val in df_sample['proto'].fillna('').astype(str):
            if val and val not in ['-', '', 'nan', 'None']:
                proto_values.add(val)
            else:
                proto_values.add('unknown')

        for val in df_sample['conn_state'].fillna('').astype(str):
            if val and val not in ['-', '', 'nan', 'None']:
                conn_state_values.add(val)
            else:
                conn_state_values.add('unknown')

    except Exception as e:
        print(f"  Warning: Could not scan {os.path.basename(file_path)}: {e}")

    return proto_values, conn_state_values


def process_single_csv(args):
    """Process one CSV file - MUST BE AT MODULE LEVEL"""
    file_path, all_categorical_values = args
    base_name = os.path.basename(file_path)
    file_key = base_name.replace('.csv', '')
    return process_csv_with_chunked_pandas(file_path, file_key, all_categorical_values)


def extract_ip_features(ip_series):
    """Extract features from IP addresses."""

    def parse_ip(ip_str):
        if pd.isna(ip_str) or ip_str in ['-', ''] or ':' in str(ip_str):
            return [0, 0, 0, 0, 0]
        try:
            octets = str(ip_str).split('.')
            if len(octets) != 4:
                return [0, 0, 0, 0, 0]
            first = int(octets[0])
            is_private = 1 if first == 10 or first == 192 or (first == 172 and 16 <= int(octets[1]) <= 31) else 0
            is_broadcast = 1 if ip_str == '255.255.255.255' else 0
            is_multicast = 1 if 224 <= first <= 239 else 0
            is_localhost = 1 if first == 127 else 0
            return [is_private, is_broadcast, is_multicast, first, is_localhost]
        except (ValueError, IndexError):
            return [0, 0, 0, 0, 0]

    ip_features = ip_series.apply(parse_ip)
    return pd.DataFrame(ip_features.tolist(), columns=Config.IP_FEATURES_BASE, index=ip_series.index)


def _clean_and_expand_labels(df):
    """Clean and expand label columns."""
    has_label_col = 'label' in df.columns and 'detailed-label' in df.columns

    if has_label_col:
        combined_str = (df['label'].astype(str).fillna('') + ' ' +
                       df['detailed-label'].astype(str).fillna('')).str.lower().str.strip()
    else:
        combined_str = pd.Series([''] * len(df))

    df['label'] = 'Benign'
    df.loc[combined_str.str.contains('malicious', na=False), 'label'] = 'Malicious'
    df['attack_type'] = 'Benign'
    df['attack_subtype'] = pd.NA
    df['malware_family'] = pd.NA

    malicious_mask = df['label'] == 'Malicious'
    if not malicious_mask.any():
        if 'detailed-label' in df.columns:
            df = df.drop(columns=['detailed-label'], errors='ignore')
        return df

    malicious_labels = combined_str[malicious_mask]

    patterns = {
        'attack_type': {
            'C&C': re.compile(r'c&c'),
            'DDoS': re.compile(r'ddos'),
            'PortScan': re.compile(r'partofahorizontalportscan|portscan'),
            'Attack': re.compile(r'attack'),
            'FileDownload': re.compile(r'filedownload')
        },
        'malware_family': {
            'Mirai': re.compile(r'mirai'),
            'Okiru': re.compile(r'okiru'),
            'Torii': re.compile(r'torii')
        },
        'attack_subtype': {
            'HeartBeat': re.compile(r'heartbeat'),
            'FileDownload': re.compile(r'filedownload')
        }
    }

    def parse_label(label_str):
        label_str = re.sub(r'\(empty\)|-|\bmalicious\b|\bbenign\b', '', label_str).strip()
        parsed = {'attack_type': set(), 'attack_subtype': set(), 'malware_family': set()}

        for category, cat_patterns in patterns.items():
            for key, pattern in cat_patterns.items():
                if pattern.search(label_str):
                    parsed[category].add(key)

        primary_type = 'Unknown'
        if parsed['malware_family']:
            primary_type = list(parsed['malware_family'])[0]
        elif 'PortScan' in parsed['attack_type']:
            primary_type = 'PortScan'
        elif 'C&C' in parsed['attack_type']:
            primary_type = 'C&C'
        elif 'DDoS' in parsed['attack_type']:
            primary_type = 'DDoS'
        elif 'FileDownload' in parsed['attack_type']:
            primary_type = 'FileDownload'
        elif 'Attack' in parsed['attack_type']:
            primary_type = 'Attack'

        subtype = ','.join(sorted(list(parsed['attack_subtype']))) if parsed['attack_subtype'] else pd.NA
        family = list(parsed['malware_family'])[0] if parsed['malware_family'] else pd.NA

        if primary_type == 'Unknown' and label_str:
            primary_type = label_str.replace(' ', '').title()

        return primary_type, subtype, family

    parsed_results = malicious_labels.apply(parse_label)
    df.loc[malicious_mask, 'attack_type'] = [res[0] for res in parsed_results]
    df.loc[malicious_mask, 'attack_subtype'] = [res[1] for res in parsed_results]
    df.loc[malicious_mask, 'malware_family'] = [res[2] for res in parsed_results]

    if 'detailed-label' in df.columns:
        df = df.drop(columns=['detailed-label'], errors='ignore')
    return df


def scan_categorical_values(csv_files):
    """OPTIMIZED: Parallel scanning of categorical values."""
    print("\n" + "=" * 70)
    print("🔍 SCANNING FILES FOR CATEGORICAL VALUES (Parallel)")
    print("=" * 70)

    t_start = time.time()

    # PARALLEL SCANNING
    all_proto_values = set()
    all_conn_state_values = set()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(scan_single_file, fp): fp for fp in csv_files}

        for future in tqdm(as_completed(futures), total=len(csv_files), desc="Scanning files"):
            proto_vals, conn_vals = future.result()
            all_proto_values.update(proto_vals)
            all_conn_state_values.update(conn_vals)

    t_elapsed = time.time() - t_start

    print(f"\n✅ Schema discovery complete ({t_elapsed:.2f}s)")
    print(f"   Found {len(all_proto_values)} unique proto values")
    print(f"   Found {len(all_conn_state_values)} unique conn_state values")

    return {
        'proto': sorted(all_proto_values),
        'conn_state': sorted(all_conn_state_values)
    }


def _preprocess_partition(df, all_categorical_values):
    """Preprocess a single partition."""

    # ==================== 1. PREVENT DATA LEAKAGE ====================
    METADATA_COLUMNS = ['Source_Folder', 'ts', 'uid', 'id.orig_h', 'id.orig_p',
                        'id.resp_h', 'id.resp_p', 'label', 'detailed-label']
    ip_columns_to_keep = ['id.orig_h', 'id.resp_h', 'id.resp_p']

    if 'label' in df.columns and 'detailed-label' in df.columns:
        df = _clean_and_expand_labels(df)

    to_drop = [col for col in METADATA_COLUMNS if col in df.columns and col not in ip_columns_to_keep]
    df = df.drop(columns=to_drop, errors='ignore')

    # ==================== 2. BASE NUMERICAL FEATURES ====================
    for col in Config.BASE_NUMERICAL_FEATURES:
        if col not in df.columns:
            df[col] = 0
            df[f'{col}_was_missing'] = 1
            continue

        df[f'{col}_was_missing'] = df[col].isna().astype(np.int8)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)

        if col in Config.SKEWED_NUMERICAL_FEATURES:
            df[col] = np.log1p(df[col])

    # ==================== 3. CATEGORICAL FEATURES ====================
    for col in Config.CATEGORICAL_LABEL_ENCODE:
        if col not in df.columns:
            df[col] = 0
            continue
        df[col] = df[col].fillna('unknown').astype(str).replace(['', '-', 'nan', 'None'], 'unknown')
        df[col] = pd.Categorical(df[col]).codes

    for col in Config.CATEGORICAL_ONE_HOT_ENCODE:
        if col not in df.columns:
            continue

        df[col] = df[col].fillna('unknown').astype(str).replace(['', '-', 'nan', 'None'], 'unknown')

        for category in all_categorical_values.get(col, []):
            df[f'{col}_{category}'] = (df[col] == category).astype(np.int8)

        df = df.drop(columns=[col])

    # ==================== 4. ADVANCED ENGINEERED FEATURES ====================
    df['is_port_23'] = (df['id.resp_p'] == 23).astype(np.int8)
    df['is_port_22'] = (df['id.resp_p'] == 22).astype(np.int8)

    df['is_S0_state'] = df.get('conn_state_S0', 0)
    df['is_telnet'] = 0
    df['is_unknown_service'] = 0

    total_bytes = df['orig_bytes'] + df['resp_bytes']
    total_packets = df['orig_pkts'] + df['resp_pkts']

    df['upload_ratio'] = df['orig_bytes'] / (total_bytes + 1e-9)
    df['bytes_per_packet'] = total_bytes / (total_packets + 1e-9)

    safe_duration = np.expm1(df['duration']).clip(lower=0.001)
    df['packet_rate'] = (total_packets / safe_duration).clip(upper=10000)

    df['is_scanning_signature'] = ((df['is_port_23'] == 1) & (df['is_S0_state'] == 1)).astype(np.int8)
    df['suspicious_score'] = df['is_port_23'] * 40 + df['is_S0_state'] * 30 + df['is_port_22'] * 25

    # ==================== 5. IP FEATURES ====================
    if 'id.resp_h' in df.columns:
        resp_ip_features = extract_ip_features(df['id.resp_h'])
        resp_ip_features.columns = ['resp_' + col for col in resp_ip_features.columns]
        df = pd.concat([df, resp_ip_features], axis=1)
        df = df.drop(columns=['id.resp_h'])

    if 'id.orig_h' in df.columns:
        orig_ip_features = extract_ip_features(df['id.orig_h'])
        orig_ip_features.columns = ['orig_' + col for col in orig_ip_features.columns]
        df = pd.concat([df, orig_ip_features], axis=1)
        df = df.drop(columns=['id.orig_h'])

    # ==================== 6. FINAL CLEANUP ====================
    df = df.drop(columns=['id.resp_p'], errors='ignore')

    return df


def process_csv_with_chunked_pandas(file_path, file_key, all_categorical_values):
    """OPTIMIZED: Larger chunks + faster I/O"""
    output_path = os.path.join(Config.ENGINEERED_SPLIT_DIR, os.path.basename(file_path))

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"\n  📄 {os.path.basename(file_path)} ({file_size_mb:.1f} MB)")

    t_start = time.time()
    first_chunk = True
    total_rows = 0

    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False, on_bad_lines='skip'):
        processed = _preprocess_partition(chunk, all_categorical_values)

        ground_truth_family = Config.FILENAME_TO_FAMILY_MAP.get(file_key, 'Unknown')
        processed[Config.FAMILY_TARGET_COL] = ground_truth_family

        if ground_truth_family == 'Benign':
            processed[Config.TARGET_COL] = 'Benign'
            processed[Config.DETAILED_TARGET_COL] = 'Benign'
        else:
            if Config.TARGET_COL not in processed.columns:
                processed[Config.TARGET_COL] = 'Malicious'
            if Config.DETAILED_TARGET_COL not in processed.columns:
                processed[Config.DETAILED_TARGET_COL] = 'Unknown'

        processed.to_csv(output_path, mode='w' if first_chunk else 'a',
                         header=first_chunk, index=False)
        first_chunk = False
        total_rows += len(processed)

    t_elapsed = time.time() - t_start
    throughput = file_size_mb / t_elapsed if t_elapsed > 0 else 0

    print(f"    ✅ {total_rows:,} rows | {t_elapsed:.1f}s | {throughput:.1f} MB/s")

    return total_rows


def build_engineered_dataset():
    """OPTIMIZED: Parallel file processing"""
    print("\n" + "=" * 70)
    print("🚀 STARTING OPTIMIZED DATASET BUILD PIPELINE")
    print("=" * 70)
    print(f"   Using {MAX_WORKERS} parallel workers")
    print(f"   Chunk size: {CHUNK_SIZE:,} rows")
    print("=" * 70)

    overall_start = time.time()

    csv_files = glob.glob(os.path.join(Config.RAW_DIR_ORIGINAL, '*.csv'))
    split_output_dir = Config.ENGINEERED_SPLIT_DIR

    if not csv_files:
        print(f"❌ Error: No CSV files found in '{Config.RAW_DIR_ORIGINAL}'")
        return

    print(f"\n📁 Found {len(csv_files)} CSV files to process")

    all_categorical_values = scan_categorical_values(csv_files)

    # ==================== STAGE 1: PARALLEL FILE PROCESSING ====================
    print("\n" + "=" * 70)
    print("🔍 STAGE 1: PROCESSING FILES (PARALLEL)")
    print("=" * 70)

    stage1_start = time.time()
    files_processed = 0
    files_skipped = 0
    total_rows = 0

    files_to_process = []
    for file_path in csv_files:
        base_name = os.path.basename(file_path)
        output_path = os.path.join(split_output_dir, base_name)

        if os.path.exists(output_path):
            files_skipped += 1
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  ⭐️ {base_name} ({file_size_mb:.1f} MB) - Already processed")
            try:
                existing_rows = sum(1 for _ in open(output_path)) - 1
                total_rows += existing_rows
            except:
                pass
        else:
            files_to_process.append(file_path)

    # PARALLEL PROCESSING
    if files_to_process:
        print(f"\n🚀 Processing {len(files_to_process)} files in parallel...")

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Create args tuples
            args_list = [(fp, all_categorical_values) for fp in files_to_process]
            futures = {executor.submit(process_single_csv, args): args for args in args_list}

            for future in as_completed(futures):
                try:
                    row_count = future.result()
                    files_processed += 1
                    total_rows += row_count
                except Exception as e:
                    print(f"    ❌ ERROR: {e}")

    stage1_time = time.time() - stage1_start

    print("\n" + "=" * 70)
    print(f"✅ STAGE 1 COMPLETE")
    print(f"   ⏱️  Time: {stage1_time:.2f}s ({stage1_time / 60:.1f} min)")
    print(f"   ✓ Processed: {files_processed} files")
    print(f"   ⭐️ Skipped: {files_skipped} files")
    print(f"   📊 Total rows: {total_rows:,}")
    print("=" * 70)

    # ==================== STAGE 2: COMBINE ALL FILES ====================
    print("\n" + "=" * 70)
    print("🔍 STAGE 2: COMBINING WITH DASK")
    print("=" * 70)

    stage2_start = time.time()
    split_files = glob.glob(os.path.join(Config.ENGINEERED_SPLIT_DIR, '*.csv'))

    if not split_files:
        print("❌ No engineered files found")
        return

    print(f"📁 Found {len(split_files)} files to combine")

    try:
        print("\n⏳ Reading with Dask...")
        ddf = dd.read_csv(
            os.path.join(Config.ENGINEERED_SPLIT_DIR, '*.csv'),
            blocksize='256MB',
            assume_missing=True,
            dtype={'attack_subtype': 'object'}
        )

        if Config.SAMPLE_SIZE:
            print(f"⏳ Sampling {Config.SAMPLE_SIZE:,} rows...")
            total_est = len(ddf)
            sample_frac = min(1.0, Config.SAMPLE_SIZE / total_est)
            ddf = ddf.sample(frac=sample_frac, random_state=Config.RANDOM_STATE)

        print("\n⏳ Building feature list...")
        final_feature_list = (
                Config.BASE_NUMERICAL_FEATURES +
                [f'{c}_was_missing' for c in Config.BASE_NUMERICAL_FEATURES] +
                Config.CATEGORICAL_LABEL_ENCODE +
                Config.ENGINEERED_FEATURES +
                [f'orig_{c}' for c in Config.IP_FEATURES_BASE] +
                [f'resp_{c}' for c in Config.IP_FEATURES_BASE]
        )

        first_row = ddf.head(1)
        one_hot_cols = [c for c in first_row.columns if
                        any(c.startswith(p + '_') for p in Config.CATEGORICAL_ONE_HOT_ENCODE)]
        final_feature_list.extend(one_hot_cols)

        target_cols = [Config.TARGET_COL, Config.DETAILED_TARGET_COL,
                       Config.FAMILY_TARGET_COL, 'attack_subtype']

        for col in final_feature_list + target_cols:
            if col not in ddf.columns:
                ddf[col] = 0 if col in final_feature_list else 'Unknown'

        final_ddf = ddf[final_feature_list + target_cols]

        print("\n⏳ Saving Parquet (streaming)...")
        t_parquet_start = time.time()

        final_ddf.to_parquet(
            Config.ENGINEERED_DATA_PATH,
            compression='snappy',
            engine='pyarrow',
            write_index=False
        )

        t_parquet = time.time() - t_parquet_start
        parquet_size = os.path.getsize(Config.ENGINEERED_DATA_PATH) / (1024 * 1024)
        print(f"✓ Parquet saved ({t_parquet:.2f}s) | {parquet_size:.1f} MB")

        print("\n⏳ Saving CSV...")
        ddf_csv = dd.read_parquet(Config.ENGINEERED_DATA_PATH)
        ddf_csv.to_csv(Config.ENGINEERED_DATA_PATH_CSV, single_file=True, index=False)
        print("✓ CSV saved")

        joblib.dump(final_feature_list, Config.FEATURE_LIST_PATH)
        print(f"✓ Feature list saved")

        stage2_time = time.time() - stage2_start
        total_time = time.time() - overall_start

        print("\n" + "=" * 70)
        print("🎉 PIPELINE COMPLETE")
        print("=" * 70)
        print(f"⏱️  Stage 1: {stage1_time:.2f}s ({stage1_time / 60:.1f} min)")
        print(f"⏱️  Stage 2: {stage2_time:.2f}s ({stage2_time / 60:.1f} min)")
        print(f"⏱️  Total: {total_time:.2f}s ({total_time / 60:.1f} min)")
        print(f"📊 Features: {len(final_feature_list)}")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Stage 2 failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    build_engineered_dataset()