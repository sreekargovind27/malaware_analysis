"""
OPTIMIZED: Parallel processing + faster I/O + better chunking
UPDATED: Implemented a two-pass "budgeted" sampling strategy to preserve small files.
"""
import glob
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config

# ==================== CONSTANTS ====================
CHUNK_SIZE = 1_000_000
MAX_WORKERS = min(16, cpu_count() - 1)
# Any file with fewer rows than this will be included in its entirety in the final sample.
SMALL_FILE_THRESHOLD = 5_000_000


# ==================== MODULE-LEVEL FUNCTIONS FOR PICKLING ====================
def scan_single_file(file_path):
    """Scan one file for categorical values."""
    proto_values, conn_state_values = set(), set()
    try:
        # Added comment='#' to handle files that might have commented header lines
        df_sample = pd.read_csv(file_path, usecols=['proto', 'conn_state'], low_memory=False, on_bad_lines='skip', comment='#')
        proto_values.update(df_sample['proto'].fillna('unknown').astype(str).replace(['-', '', 'nan', 'None'], 'unknown'))
        conn_state_values.update(df_sample['conn_state'].fillna('unknown').astype(str).replace(['-', '', 'nan', 'None'], 'unknown'))
    except Exception as e:
        print(f"  Warning: Could not scan {os.path.basename(file_path)}: {e}")
    return proto_values, conn_state_values

# ============================================================================
# THIS FUNCTION WAS MISSING. IT HAS BEEN ADDED BACK.
# ============================================================================
def scan_categorical_values(csv_files):
    """OPTIMIZED: Parallel scanning of categorical values."""
    print("\n" + "=" * 70)
    print("🔍 SCANNING FILES FOR CATEGORICAL VALUES (Parallel)")
    print("=" * 70)
    t_start = time.time()
    all_proto_values, all_conn_state_values = set(), set()
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
    return {'proto': sorted(all_proto_values), 'conn_state': sorted(all_conn_state_values)}
# ============================================================================
# END OF ADDED FUNCTION
# ============================================================================


def process_single_csv(args):
    """Process one CSV file."""
    file_path, all_categorical_values = args
    base_name = os.path.basename(file_path)
    file_key = base_name.replace('.csv', '')
    return process_csv_with_chunked_pandas(file_path, file_key, all_categorical_values)


def extract_ip_features(ip_series):
    """Extract features from IP addresses."""
    def parse_ip(ip_str):
        if pd.isna(ip_str) or ip_str in ['-', ''] or ':' in str(ip_str): return [0, 0, 0, 0, 0]
        try:
            octets = str(ip_str).split('.')
            if len(octets) != 4: return [0, 0, 0, 0, 0]
            first = int(octets[0])
            is_private = 1 if first == 10 or first == 192 or (first == 172 and 16 <= int(octets[1]) <= 31) else 0
            is_broadcast = 1 if ip_str == '255.255.255.255' else 0
            is_multicast = 1 if 224 <= first <= 239 else 0
            is_localhost = 1 if first == 127 else 0
            return [is_private, is_broadcast, is_multicast, first, is_localhost]
        except (ValueError, IndexError): return [0, 0, 0, 0, 0]
    ip_features = ip_series.apply(parse_ip)
    return pd.DataFrame(ip_features.tolist(), columns=Config.IP_FEATURES_BASE, index=ip_series.index)


def _clean_and_expand_labels(df):
    """Clean and expand label columns."""
    has_label_col = 'label' in df.columns and 'detailed-label' in df.columns
    if has_label_col:
        combined_str = (df['label'].astype(str).fillna('') + ' ' + df['detailed-label'].astype(str).fillna('')).str.lower().str.strip()
    else:
        combined_str = pd.Series([''] * len(df))

    df['label'] = 'Benign'
    df.loc[combined_str.str.contains('malicious', na=False), 'label'] = 'Malicious'
    df['attack_type'] = 'Benign'
    df['attack_subtype'] = pd.NA
    df['malware_family'] = pd.NA

    malicious_mask = df['label'] == 'Malicious'
    if not malicious_mask.any():
        if 'detailed-label' in df.columns: df = df.drop(columns=['detailed-label'], errors='ignore')
        return df

    malicious_labels = combined_str[malicious_mask]
    patterns = {'attack_type': {'C&C': re.compile(r'c&c'), 'DDoS': re.compile(r'ddos'), 'PortScan': re.compile(r'partofahorizontalportscan|portscan'), 'Attack': re.compile(r'attack'), 'FileDownload': re.compile(r'filedownload')}, 'malware_family': {'Mirai': re.compile(r'mirai'), 'Okiru': re.compile(r'okiru'), 'Torii': re.compile(r'torii')}, 'attack_subtype': {'HeartBeat': re.compile(r'heartbeat'), 'FileDownload': re.compile(r'filedownload')}}

    def parse_label(label_str):
        label_str = re.sub(r'\(empty\)|-|\bmalicious\b|\bbenign\b', '', label_str).strip()
        parsed = {'attack_type': set(), 'attack_subtype': set(), 'malware_family': set()}
        for category, cat_patterns in patterns.items():
            for key, pattern in cat_patterns.items():
                if pattern.search(label_str): parsed[category].add(key)
        primary_type = 'Unknown'
        if parsed['malware_family']: primary_type = list(parsed['malware_family'])[0]
        elif 'PortScan' in parsed['attack_type']: primary_type = 'PortScan'
        elif 'C&C' in parsed['attack_type']: primary_type = 'C&C'
        elif 'DDoS' in parsed['attack_type']: primary_type = 'DDoS'
        elif 'FileDownload' in parsed['attack_type']: primary_type = 'FileDownload'
        elif 'Attack' in parsed['attack_type']: primary_type = 'Attack'
        subtype = ','.join(sorted(list(parsed['attack_subtype']))) if parsed['attack_subtype'] else pd.NA
        family = list(parsed['malware_family'])[0] if parsed['malware_family'] else pd.NA
        if primary_type == 'Unknown' and label_str: primary_type = label_str.replace(' ', '').title()
        return primary_type, subtype, family

    parsed_results = malicious_labels.apply(parse_label)
    df.loc[malicious_mask, 'attack_type'] = [res[0] for res in parsed_results]
    df.loc[malicious_mask, 'attack_subtype'] = [res[1] for res in parsed_results]
    df.loc[malicious_mask, 'malware_family'] = [res[2] for res in parsed_results]
    if 'detailed-label' in df.columns: df = df.drop(columns=['detailed-label'], errors='ignore')
    return df


def _preprocess_partition(df, all_categorical_values):
    """Preprocess a single partition."""
    METADATA_COLUMNS = ['Source_Folder', 'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'label', 'detailed-label']
    ip_columns_to_keep = ['id.orig_h', 'id.resp_h', 'id.resp_p']
    if 'label' in df.columns and 'detailed-label' in df.columns: df = _clean_and_expand_labels(df)

    # Drop metadata, but KEEP local_orig/local_resp if they exist, to be processed later
    to_drop = [col for col in METADATA_COLUMNS if col in df.columns and col not in ip_columns_to_keep]
    df = df.drop(columns=to_drop, errors='ignore')

    # Process local_orig/resp if they exist in the chunk
    if 'local_orig' in df.columns:
        df['local_orig'] = pd.to_numeric(df['local_orig'], errors='coerce').fillna(0).astype(np.int8)
    if 'local_resp' in df.columns:
        df['local_resp'] = pd.to_numeric(df['local_resp'], errors='coerce').fillna(0).astype(np.int8)

    for col in Config.BASE_NUMERICAL_FEATURES:
        if col not in df.columns:
            df[col] = 0; df[f'{col}_was_missing'] = 1; continue
        df[f'{col}_was_missing'] = df[col].isna().astype(np.int8)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)
        if col in Config.SKEWED_NUMERICAL_FEATURES: df[col] = np.log1p(df[col])
    for col in Config.CATEGORICAL_LABEL_ENCODE:
        if col not in df.columns: df[col] = 0; continue
        df[col] = df[col].fillna('unknown').astype(str).replace(['', '-', 'nan', 'None'], 'unknown')
        df[col] = pd.Categorical(df[col]).codes.astype(np.int32)
    for col in Config.CATEGORICAL_ONE_HOT_ENCODE:
        if col not in df.columns: continue
        df[col] = df[col].fillna('unknown').astype(str).replace(['', '-', 'nan', 'None'], 'unknown')
        for category in all_categorical_values.get(col, []): df[f'{col}_{category}'] = (df[col] == category).astype(np.int8)
        df = df.drop(columns=[col])
    df['is_port_23'] = (df['id.resp_p'] == 23).astype(np.int8); df['is_port_22'] = (df['id.resp_p'] == 22).astype(np.int8)
    df['is_S0_state'] = df.get('conn_state_S0', 0); df['is_telnet'] = 0; df['is_unknown_service'] = 0
    total_bytes = df['orig_bytes'] + df['resp_bytes']; total_packets = df['orig_pkts'] + df['resp_pkts']
    df['upload_ratio'] = df['orig_bytes'] / (total_bytes + 1e-9); df['bytes_per_packet'] = total_bytes / (total_packets + 1e-9)
    safe_duration = np.expm1(df['duration']).clip(lower=0.001); df['packet_rate'] = (total_packets / safe_duration).clip(upper=10000)
    df['is_scanning_signature'] = ((df['is_port_23'] == 1) & (df['is_S0_state'] == 1)).astype(np.int8)
    df['suspicious_score'] = df['is_port_23'] * 40 + df['is_S0_state'] * 30 + df['is_port_22'] * 25
    if 'id.resp_h' in df.columns:
        resp_ip_features = extract_ip_features(df['id.resp_h']); resp_ip_features.columns = ['resp_' + col for col in resp_ip_features.columns]
        df = pd.concat([df, resp_ip_features], axis=1); df = df.drop(columns=['id.resp_h'])
    if 'id.orig_h' in df.columns:
        orig_ip_features = extract_ip_features(df['id.orig_h']); orig_ip_features.columns = ['orig_' + col for col in orig_ip_features.columns]
        df = pd.concat([df, orig_ip_features], axis=1); df = df.drop(columns=['id.orig_h'])
    df = df.drop(columns=['id.resp_p'], errors='ignore')
    return df


def process_csv_with_chunked_pandas(file_path, file_key, all_categorical_values):
    """OPTIMIZED: Write Parquet instead of CSV"""
    output_path = os.path.join(Config.ENGINEERED_SPLIT_DIR, os.path.basename(file_path).replace('.csv', '.parquet'))
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"\n  📄 {os.path.basename(file_path)} ({file_size_mb:.1f} MB)")
    t_start = time.time(); chunks_list = []; total_rows = 0
    # Added comment='#' to handle files that might have commented header lines
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False, on_bad_lines='skip', comment='#'):
        processed = _preprocess_partition(chunk, all_categorical_values)
        ground_truth_family = Config.FILENAME_TO_FAMILY_MAP.get(file_key, 'Unknown')
        processed[Config.FAMILY_TARGET_COL] = ground_truth_family
        if ground_truth_family == 'Benign':
            processed[Config.TARGET_COL] = 'Benign'; processed[Config.DETAILED_TARGET_COL] = 'Benign'
        else:
            if Config.TARGET_COL not in processed.columns: processed[Config.TARGET_COL] = 'Malicious'
            if Config.DETAILED_TARGET_COL not in processed.columns: processed[Config.DETAILED_TARGET_COL] = 'Unknown'
        chunks_list.append(processed); total_rows += len(processed)
    if chunks_list:
        final_df = pd.concat(chunks_list, ignore_index=True)
        final_df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    t_elapsed = time.time() - t_start
    throughput = file_size_mb / t_elapsed if t_elapsed > 0 else 0
    print(f"    ✅ {total_rows:,} rows | {t_elapsed:.1f}s | {throughput:.1f} MB/s")
    return total_rows


def build_engineered_dataset():
    """Builds the final dataset with a smart, budgeted sampling strategy."""
    print("\n" + "=" * 70)
    print("🚀 STARTING OPTIMIZED DATASET BUILD PIPELINE")
    print("=" * 70)
    print(f"   Using {MAX_WORKERS} parallel workers")
    print(f"   Chunk size: {CHUNK_SIZE:,} rows")
    print(f"   Target sample size: {Config.SAMPLE_SIZE:,} rows")
    print(f"   Small file threshold: {SMALL_FILE_THRESHOLD:,} rows")
    print("=" * 70)

    overall_start = time.time()
    csv_files = glob.glob(os.path.join(Config.RAW_DIR_ORIGINAL, '*.csv'))
    if not csv_files:
        print(f"❌ Error: No CSV files found in '{Config.RAW_DIR_ORIGINAL}'"); return

    print(f"\n📁 Found {len(csv_files)} CSV files to process")
    all_categorical_values = scan_categorical_values(csv_files)

    # ==================== STAGE 1: PARALLEL FILE PROCESSING ====================
    print("\n" + "=" * 70)
    print("🔍 STAGE 1: PROCESSING FILES (PARALLEL)")
    print("=" * 70)
    stage1_start = time.time(); files_processed, files_skipped = 0, 0
    files_to_process = []
    for file_path in csv_files:
        output_path = os.path.join(Config.ENGINEERED_SPLIT_DIR, os.path.basename(file_path).replace('.csv', '.parquet'))
        if os.path.exists(output_path): files_skipped += 1
        else: files_to_process.append(file_path)

    if files_to_process:
        print(f"\n🚀 Processing {len(files_to_process)} new files in parallel...")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            args_list = [(fp, all_categorical_values) for fp in files_to_process]
            futures = {executor.submit(process_single_csv, args): args for args in args_list}
            for future in as_completed(futures):
                try: future.result(); files_processed += 1
                except Exception as e: print(f"    ❌ ERROR: {e}")

    stage1_time = time.time() - stage1_start
    print("\n" + "=" * 70)
    print(f"✅ STAGE 1 COMPLETE")
    print(f"   ⏱️  Time: {stage1_time:.2f}s ({stage1_time / 60:.1f} min)")
    print(f"   ✓ Processed: {files_processed} files")
    print(f"   ⭐️ Skipped: {files_skipped} files")
    print("=" * 70)

    # ==================== STAGE 2: BUDGETED SAMPLING ====================
    print("\n" + "=" * 70)
    print("🔍 STAGE 2: COMBINING WITH BUDGETED SAMPLING")
    print("=" * 70)
    stage2_start = time.time()
    split_files = glob.glob(os.path.join(Config.ENGINEERED_SPLIT_DIR, '*.parquet'))
    if not split_files: print("❌ No engineered files found"); return

    print(f"📁 Found {len(split_files)} processed files to sample from")

    # Pass 1: Identify small/large files and calculate budget
    small_files_to_keep, large_files_to_sample = [], []
    rows_from_small_files, total_rows_in_large_files = 0, 0

    print("⏳ Pass 1: Classifying files and calculating row counts...")
    for pq_file in tqdm(split_files, desc="Analyzing files"):
        # Read only one column for speed to get the row count
        row_count = len(pd.read_parquet(pq_file, columns=['label']))
        if row_count < SMALL_FILE_THRESHOLD:
            small_files_to_keep.append(pq_file)
            rows_from_small_files += row_count
        else:
            large_files_to_sample.append((pq_file, row_count))
            total_rows_in_large_files += row_count

    print(f"✓ Found {len(small_files_to_keep)} small files (Total: {rows_from_small_files:,} rows) to be fully included.")
    print(f"✓ Found {len(large_files_to_sample)} large files (Total: {total_rows_in_large_files:,} rows) to be sampled from.")

    remaining_rows_to_sample = Config.SAMPLE_SIZE - rows_from_small_files

    # Pass 2: Load/sample files based on the budget
    all_samples = []
    if remaining_rows_to_sample <= 0:
        print("⚠️ Warning: Small files alone meet or exceed the sample size. Sampling only from small files to meet target.")
        temp_df = pd.concat([pd.read_parquet(f) for f in small_files_to_keep], ignore_index=True)
        all_samples.append(temp_df.sample(n=Config.SAMPLE_SIZE, random_state=Config.RANDOM_STATE))
    else:
        print(f"⏳ Pass 2: Loading all small files and sampling {remaining_rows_to_sample:,} rows from large files...")
        # Load 100% of all small files
        for f in tqdm(small_files_to_keep, desc="Loading small files"):
            all_samples.append(pd.read_parquet(f))

        # Sample proportionally from large files to fill the budget
        if total_rows_in_large_files > 0:
            large_file_frac = remaining_rows_to_sample / total_rows_in_large_files
            for pq_file, row_count in tqdm(large_files_to_sample, desc="Sampling large files"):
                n_sample = int(row_count * large_file_frac)
                if n_sample > 0:
                    df_chunk = pd.read_parquet(pq_file)
                    all_samples.append(df_chunk.sample(n=n_sample, random_state=Config.RANDOM_STATE))

    print("⏳ Concatenating all samples...")
    final_df = pd.concat(all_samples, ignore_index=True)
    del all_samples
    print(f"✓ Final dataset created with {len(final_df):,} rows.")

    # FIX: Explicitly correct the data types for local_orig and local_resp AFTER concatenation.
    # This prevents them from being upcast to 'object' dtype, which causes errors in LightGBM.
    if 'local_orig' in final_df.columns:
        final_df['local_orig'] = pd.to_numeric(final_df['local_orig'], errors='coerce').fillna(0).astype(np.int8)
    if 'local_resp' in final_df.columns:
        final_df['local_resp'] = pd.to_numeric(final_df['local_resp'], errors='coerce').fillna(0).astype(np.int8)

    # Final processing and saving
    print("\n⏳ Building final feature list...")
    target_cols = [Config.TARGET_COL, Config.DETAILED_TARGET_COL, Config.FAMILY_TARGET_COL, 'attack_subtype']
    final_feature_list = [col for col in final_df.columns if col not in target_cols]
    final_df = final_df[final_feature_list + target_cols] # Ensure column order
    joblib.dump(final_feature_list, Config.FEATURE_LIST_PATH)
    print(f"✓ Feature list saved with {len(final_feature_list)} features.")

    print("\n⏳ Saving Parquet...")
    final_df.to_parquet(Config.ENGINEERED_DATA_PATH, compression='snappy', index=False)
    print(f"✓ Parquet saved to {Config.ENGINEERED_DATA_PATH}")

    print("\n⏳ Saving CSV...")
    final_df.to_csv(Config.ENGINEERED_DATA_PATH_CSV, index=False)
    print(f"✓ CSV saved to {Config.ENGINEERED_DATA_PATH_CSV}")

    stage2_time = time.time() - stage2_start
    total_time = time.time() - overall_start
    print("\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETE")
    print("=" * 70)
    print(f"⏱️  Stage 1: {stage1_time:.2f}s ({stage1_time / 60:.1f} min)")
    print(f"⏱️  Stage 2: {stage2_time:.2f}s ({stage2_time / 60:.1f} min)")
    print(f"⏱️  Total: {total_time:.2f}s ({total_time / 60:.1f} min)")
    print(f"📊 Final rows: {len(final_df):,}")
    print(f"📊 Features: {len(final_feature_list)}")
    print("=" * 70)


if __name__ == "__main__":
    build_engineered_dataset()