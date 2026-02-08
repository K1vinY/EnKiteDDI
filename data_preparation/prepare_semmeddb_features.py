
import os
import sys
import io
import pandas as pd
import pickle
import numpy as np
import argparse
import re
from collections import defaultdict, Counter
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r'\s*(hydrochloride|sodium|potassium|acetate|sulfate|mesylate|maleate|tartrate|citrate|fumarate|succinate|chloride|bromide)$', '', name)
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def step0_build_drkg_mapping(drkg_path, output_dir):
    print("\n" + "="*70)
    print("Step 0: Building DRKG Drug Mapping Table")
    print("="*70)
    
    drkg_compounds = set()
    total_lines = 0
    
    print("  Extracting Compound entities from DRKG...")
    with open(drkg_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            parts = line.strip().split('\t')
            if len(parts) == 3:
                head, rel, tail = parts
                if head.startswith('Compound::DB'):
                    drkg_compounds.add(head.replace('Compound::', ''))
                if tail.startswith('Compound::DB'):
                    drkg_compounds.add(tail.replace('Compound::', ''))
            
            if total_lines % 1000000 == 0:
                print(f"    Processed {total_lines:,} lines...")
    
    print(f"  Found {len(drkg_compounds):,} unique DrugBank compounds")
    
    db1 = pd.read_csv(os.path.join(output_dir, 'db1_drugs.csv'))
    db2 = pd.read_csv(os.path.join(output_dir, 'db2_drugs.csv'))
    
    name_to_id = {}
    id_to_name = {}
    
    for _, row in db1.iterrows():
        name_lower = row['name'].lower().strip()
        name_to_id[name_lower] = row['id']
        id_to_name[row['id']] = row['name']
    
    for _, row in db2.iterrows():
        name_lower = row['name'].lower().strip()
        if name_lower not in name_to_id:
            name_to_id[name_lower] = row['id']
        if row['id'] not in id_to_name:
            id_to_name[row['id']] = row['name']
    
    mapping = {
        'drkg_compounds': list(drkg_compounds),
        'name_to_id': name_to_id,
        'id_to_name': id_to_name,
    }
    
    mapping_path = os.path.join(output_dir, 'drkg_drug_mapping.pkl')
    with open(mapping_path, 'wb') as f:
        pickle.dump(mapping, f)
    
    print(f"  Saved mapping to: {mapping_path}")
    print(f"  Name mappings: {len(name_to_id):,}")
    return mapping_path


def step0b_update_mapping(vocab_path, mapping_path, output_dir):
    print("\n" + "="*70)
    print("Step 0b: Updating Drug Mapping with DrugBank Vocabulary")
    print("="*70)
    
    with open(mapping_path, 'rb') as f:
        mapping = pickle.load(f)
    
    name_to_id = mapping['name_to_id'].copy()
    drkg_compounds = mapping['drkg_compounds']
    
    print("  Loading DrugBank vocabulary...")
    vocab = pd.read_csv(vocab_path, encoding='utf-8')
    print(f"  Total vocabulary entries: {len(vocab):,}")
    
    added_count = 0
    for idx, row in vocab.iterrows():
        drug_id = str(row.iloc[0]).strip()
        if not drug_id.startswith('DB'):
            continue
        
        names = []
        for col in vocab.columns[1:]:
            val = str(row[col])
            if pd.notna(val) and val.strip() and val != 'nan':
                names.extend([n.strip() for n in val.split('|') if n.strip()])
        
        for name in names:
            normalized = normalize_name(name)
            if normalized and drug_id in drkg_compounds:
                if normalized not in name_to_id:
                    name_to_id[normalized] = drug_id
                    added_count += 1
        
        if (idx + 1) % 10000 == 0:
            print(f"    Processed {idx+1:,} / {len(vocab):,} entries... (added: {added_count:,})")
    
    mapping['name_to_id'] = name_to_id
    mapping['updated_at'] = datetime.now().isoformat()
    
    updated_path = os.path.join(output_dir, 'drkg_drug_mapping.pkl')
    with open(updated_path, 'wb') as f:
        pickle.dump(mapping, f)
    
    print(f"  Added/updated {added_count:,} name mappings")
    print(f"  Total mappings: {len(name_to_id):,}")
    return updated_path


def step1_filter_semmeddb(semmeddb_path, output_dir):
    print("\n" + "="*70)
    print("Step 1: Filtering DDI Relations from SemMedDB")
    print("="*70)
    
    DDI_PREDICATES = {
        'INTERACTS_WITH', 'INHIBITS', 'STIMULATES', 'AFFECTS', 'AUGMENTS',
        'DISRUPTS', 'PREVENTS', 'CAUSES', 'PREDISPOSES', 'PRODUCES',
        'TREATS', 'COEXISTS_WITH'
    }
    
    DRUG_SEMTYPES = {'phsu', 'orch', 'phsf', 'antb', 'clnd'}
    
    print(f"  Reading SemMedDB file: {semmeddb_path}")
    print("  This may take several minutes for large files...")
    
    chunks = []
    chunk_size = 1000000
    
    COLUMN_NAMES = [
        'PREDICATION_ID', 'SENTENCE_ID', 'PMID', 'PREDICATE',
        'SUBJECT_CUI', 'SUBJECT_NAME', 'SUBJECT_SEMTYPE', 'SUBJECT_NOVELTY',
        'OBJECT_CUI', 'OBJECT_NAME', 'OBJECT_SEMTYPE', 'OBJECT_NOVELTY',
        'COL12', 'COL13', 'COL14'
    ]
    
    for chunk in pd.read_csv(semmeddb_path, chunksize=chunk_size, header=None, 
                             names=COLUMN_NAMES, low_memory=False, encoding='latin-1'):
        mask = chunk['PREDICATE'].isin(DDI_PREDICATES)
        chunk_filtered = chunk[mask].copy()
        
        if len(chunk_filtered) > 0:
            subj_semtype = chunk_filtered['SUBJECT_SEMTYPE'].astype(str).str.lower()
            obj_semtype = chunk_filtered['OBJECT_SEMTYPE'].astype(str).str.lower()
            drug_mask = (subj_semtype.isin(DRUG_SEMTYPES)) & (obj_semtype.isin(DRUG_SEMTYPES))
            chunk_filtered = chunk_filtered[drug_mask]
            
            mask_self = chunk_filtered['SUBJECT_NAME'].str.lower() != chunk_filtered['OBJECT_NAME'].str.lower()
            chunk_filtered = chunk_filtered[mask_self]
        
        if len(chunk_filtered) > 0:
            chunks.append(chunk_filtered)
        
        print(f"    Processed chunk, filtered: {len(chunk_filtered):,} rows")
    
    if not chunks:
        print("  Error: No DDI relations found!")
        return None
    
    filtered_df = pd.concat(chunks, ignore_index=True)
    filtered_df = filtered_df.drop_duplicates(subset=['SUBJECT_NAME', 'PREDICATE', 'OBJECT_NAME'])
    
    output_path = os.path.join(output_dir, 'semmeddb_ddi_filtered.csv')
    filtered_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"  Saved filtered DDI relations: {len(filtered_df):,}")
    print(f"  Output: {output_path}")
    return output_path


def build_feature_vector(feat, max_count, max_unique_pmids, variant='8dim'):
    count = feat['total_count']
    normalized_count = count / max_count if max_count > 0 else 0.0
    
    unique_pmids_count = len(feat['pmids'])
    normalized_unique_pmids = unique_pmids_count / max_unique_pmids if max_unique_pmids > 0 else 0.0
    
    if variant == '8dim':
        feature_vec = [
            feat['has_relation'],
            *feat['relation_types'],
            normalized_count
        ]
    elif variant == '9dim':
        feature_vec = [
            feat['has_relation'],
            *feat['relation_types'],
            normalized_count,
            normalized_unique_pmids
        ]
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    return np.array(feature_vec, dtype=np.float32)


def step4_build_features(filtered_path, mapping_path, output_dir, variant='8dim', db='db1'):
    print("\n" + "="*70)
    print(f"Step 4: Building SemMedDB Features ({variant}) for {db.upper()}")
    print("="*70)
    
    with open(mapping_path, 'rb') as f:
        mapping = pickle.load(f)
    
    name_to_id = mapping['name_to_id']
    drkg_compounds = mapping['drkg_compounds']
    
    RELATION_TYPES = [
        'INTERACTS_WITH', 'INHIBITS', 'STIMULATES',
        'COEXISTS_WITH', 'PRODUCES', 'AFFECTS'
    ]
    rel_type_to_idx = {rel: i for i, rel in enumerate(RELATION_TYPES)}
    
    print("  Processing SemMedDB DDI...")
    
    drug_pair_features = defaultdict(lambda: {
        'has_relation': 0,
        'relation_types': [0] * len(RELATION_TYPES),
        'total_count': 0,
        'pmids': set()
    })
    
    semmed = pd.read_csv(filtered_path)
    print(f"  Total SemMedDB DDI: {len(semmed):,}")
    
    matched_count = 0
    
    for idx, row in semmed.iterrows():
        subj_name = str(row['SUBJECT_NAME']).lower().strip()
        obj_name = str(row['OBJECT_NAME']).lower().strip()
        predicate = row['PREDICATE']
        
        subj_id = name_to_id.get(subj_name) or name_to_id.get(normalize_name(subj_name))
        obj_id = name_to_id.get(obj_name) or name_to_id.get(normalize_name(obj_name))
        
        if subj_id and obj_id and subj_id in drkg_compounds and obj_id in drkg_compounds:
            pair = tuple(sorted([subj_id, obj_id]))
            
            drug_pair_features[pair]['has_relation'] = 1
            drug_pair_features[pair]['total_count'] += 1
            
            if predicate in rel_type_to_idx:
                drug_pair_features[pair]['relation_types'][rel_type_to_idx[predicate]] = 1
            
            if pd.notna(row.get('PMID')):
                pmid = str(row['PMID']).strip()
                if pmid:
                    drug_pair_features[pair]['pmids'].add(pmid)
            
            matched_count += 1
        
        if (idx + 1) % 500000 == 0:
            print(f"    Processed {idx+1:,} / {len(semmed):,} rows... (matched: {matched_count:,})")
    
    print(f"  Matched drug pairs: {len(drug_pair_features):,}")
    
    print("  Calculating normalization values...")
    max_count = max([feat['total_count'] for feat in drug_pair_features.values()]) if drug_pair_features else 1
    max_unique_pmids = max([len(feat['pmids']) for feat in drug_pair_features.values()]) if drug_pair_features else 1
    print(f"    Max count: {max_count:,}")
    print(f"    Max unique PMIDs: {max_unique_pmids:,}")
    
    if db == 'db1':
        db_file = os.path.join(output_dir, 'db1_drugs.csv')
    else:
        db_file = os.path.join(output_dir, 'db2_drugs.csv')
    
    if os.path.exists(db_file):
        print(f"  Loading {db.upper()} drug mapping...")
        db_drugs = pd.read_csv(db_file)
        if db == 'db1':
            db_id_to_name = dict(zip(db_drugs['id'], db_drugs['name']))
        else:
            db_ids = set(db_drugs['id'])
        print(f"    {db.upper()} drugs: {len(db_drugs):,}")
    else:
        print(f"  Warning: {db_file} not found, using all matched pairs")
        db_id_to_name = {}
        db_ids = set()
    
    print(f"  Generating {variant} features...")
    features = {}
    default_feature = np.zeros(int(variant.replace('dim', '')), dtype=np.float32)
    
    if db == 'db1' and db_id_to_name:
        db_ids = set(db_id_to_name.keys())
        for pair in drug_pair_features:
            if pair[0] in db_ids and pair[1] in db_ids:
                feat = drug_pair_features[pair]
                id1, id2 = pair
                name1, name2 = db_id_to_name[id1], db_id_to_name[id2]
                name_pair = tuple(sorted([name1, name2]))
                features[name_pair] = build_feature_vector(feat, max_count, max_unique_pmids, variant)
    elif db == 'db2' and db_ids:
        for pair, feat in drug_pair_features.items():
            id1, id2 = pair
            if id1 in db_ids and id2 in db_ids:
                features[pair] = build_feature_vector(feat, max_count, max_unique_pmids, variant)
    else:
        for pair, feat in drug_pair_features.items():
            features[pair] = build_feature_vector(feat, max_count, max_unique_pmids, variant)
    
    output_data = {
        'features': features,
        'relation_types': RELATION_TYPES,
        'feature_dim': int(variant.replace('dim', '')),
        'max_count': max_count,
        'max_unique_pmids': max_unique_pmids,
        'default': default_feature,
        'variant': variant,
        'statistics': {
            'total_pairs': len(features),
            'pairs_with_relation': sum(1 for v in features.values() if v[0] > 0),
            'matched_semmeddb_triples': matched_count
        }
    }
    
    output_filename = f'semmeddb_features_{db}_{variant}.pkl'
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"  Saved {variant} features: {len(features):,} pairs")
    print(f"  Output: {output_path}")
    print(f"  File size: {os.path.getsize(output_path)/1024/1024:.2f} MB")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Prepare SemMedDB Features for EnKiteDDI')
    parser.add_argument('--variant', type=str, default='both', 
                       choices=['8dim', '9dim', 'both'],
                       help='Feature dimension variant (default: both)')
    parser.add_argument('--db', type=str, default='both',
                       choices=['db1', 'db2', 'both'],
                       help='Dataset to process (default: both)')
    parser.add_argument('--data-dir', type=str, default='../../data',
                       help='Data directory path (default: ../../data)')
    parser.add_argument('--skip-steps', type=str, nargs='+',
                       choices=['0', '0b', '1', '4'],
                       help='Skip specific steps (e.g., --skip-steps 0 0b if mapping already exists)')
    
    args = parser.parse_args()
    
    skip_steps = set(args.skip_steps) if args.skip_steps else set()
    
    print("="*70)
    print("SemMedDB Features Preparation Pipeline")
    print("="*70)
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    data_dir = args.data_dir
    
    drkg_path = os.path.join(data_dir, 'drkg.tsv')
    vocab_path = os.path.join(data_dir, 'drugbank vocabulary.csv')
    
    mapping_path = os.path.join(data_dir, 'drkg_drug_mapping.pkl')
    if '0' not in skip_steps:
        if not os.path.exists(drkg_path):
            print(f"Error: DRKG file not found: {drkg_path}")
            return
        mapping_path = step0_build_drkg_mapping(drkg_path, data_dir)
    else:
        print("\n[Skipped] Step 0: Building DRKG mapping")
    
    if '0b' not in skip_steps:
        if not os.path.exists(vocab_path):
            print(f"Warning: DrugBank vocabulary not found: {vocab_path}")
            print("  Skipping Step 0b...")
        else:
            mapping_path = step0b_update_mapping(vocab_path, mapping_path, data_dir)
    else:
        print("\n[Skipped] Step 0b: Updating mapping with vocabulary")
    
    filtered_path = os.path.join(data_dir, 'semmeddb_ddi_filtered.csv')
    if '1' not in skip_steps:
        semmeddb_files = [
            os.path.join(data_dir, 'semmedVER43_2024_R_PREDICATION.csv'),
            os.path.join(data_dir, 'semmedVER43_2024_R_PREDICATION.23327.csv'),
        ]
        semmeddb_path = None
        for f in semmeddb_files:
            if os.path.exists(f):
                semmeddb_path = f
                break
        
        if not semmeddb_path:
            print("Error: SemMedDB file not found!")
            print("  Please download SemMedDB from https://lhncbc.nlm.nih.gov/ii/tools/SemRep_SemMedDB_SKR/SemMedDB.html")
            print("  And place the PREDICATION file in the data directory")
            return
        
        filtered_path = step1_filter_semmeddb(semmeddb_path, data_dir)
    else:
        print("\n[Skipped] Step 1: Filtering SemMedDB")
    
    if '4' not in skip_steps:
        if not os.path.exists(filtered_path):
            print(f"Error: Filtered SemMedDB not found: {filtered_path}")
            return
        
        variants = ['8dim', '9dim'] if args.variant == 'both' else [args.variant]
        dbs = ['db1', 'db2'] if args.db == 'both' else [args.db]
        
        for variant in variants:
            for db in dbs:
                step4_build_features(filtered_path, mapping_path, data_dir, variant, db)
    else:
        print("\n[Skipped] Step 4: Building features")
    
    end_time = datetime.now()
    duration = end_time - start_time
    print("\n" + "="*70)
    print("Pipeline completed successfully!")
    print("="*70)
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")


if __name__ == "__main__":
    main()


