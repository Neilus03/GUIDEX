import json
import ast
import glob
import os
import argparse
import re
from collections import defaultdict
from tqdm import tqdm
import numpy as np # For calculating average safely

def extract_class_names_from_ast(node):
    """
    Recursively extracts class names (entity types) being instantiated from an AST node.
    Looks for ast.Call nodes where the function is a simple name (ast.Name).
    """
    # (Implementation remains the same as before)
    class_names = set()
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            class_names.add(node.func.id)
            for arg in node.args:
                class_names.update(extract_class_names_from_ast(arg))
            for kw in node.keywords:
                class_names.update(extract_class_names_from_ast(kw.value))
        else:
            pass
    elif isinstance(node, (ast.List, ast.Tuple)):
        for element in node.elts:
            class_names.update(extract_class_names_from_ast(element))
    elif isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
             if key is not None:
                 class_names.update(extract_class_names_from_ast(key))
             class_names.update(extract_class_names_from_ast(value))
    return class_names

def extract_entity_types_from_file(filepath):
    """
    Reads a jsonl file and extracts unique entity type names (class names).
    Returns a tuple: (set_of_entity_types, processing_report_dict_short)
    """
    entity_types = set()
    filename = os.path.basename(filepath)
    print(f"Processing: {filename} for entity types")
    processed_lines = 0
    skipped_lines_json = 0
    skipped_lines_ast = 0
    skipped_lines_other = 0

    try:
        try:
            with open(filepath, 'r', encoding='utf-8') as f_count:
                num_lines = sum(1 for _ in f_count)
        except Exception:
            num_lines = None

        with open(filepath, 'r', encoding='utf-8') as f:
            pbar = tqdm(f, total=num_lines, desc=f"Reading {filename.split('.')[0]}", leave=False)
            for line_num, line in enumerate(pbar, 1):
                processed_lines += 1
                try:
                    data = json.loads(line)
                    labels_str = data.get('labels')

                    if labels_str and isinstance(labels_str, str) and labels_str.strip() and labels_str.strip() != '[]':
                        try:
                            tree = ast.parse(labels_str.strip(), mode='eval')
                            if isinstance(tree, ast.Expression):
                                entity_types.update(extract_class_names_from_ast(tree.body))
                        except SyntaxError:
                            skipped_lines_ast += 1
                        except Exception:
                            skipped_lines_ast += 1
                    elif labels_str and not isinstance(labels_str, str):
                         skipped_lines_other += 1

                except json.JSONDecodeError:
                    skipped_lines_json += 1
                except Exception:
                     skipped_lines_other += 1

                pbar.set_postfix({
                    "types": len(entity_types),
                    "skip_json": skipped_lines_json,
                    "skip_ast": skipped_lines_ast,
                    "skip_other": skipped_lines_other
                    }, refresh=False)

    except FileNotFoundError:
        error_msg = f"File not found: {filepath}"
        print(f"Error: {error_msg}")
        return set(), {'error': error_msg}
    except Exception as e:
        error_msg = f"General error processing file {filepath}: {e}"
        print(f"Error: {error_msg}")
        return set(), {'error': error_msg}

    skipped_counts = {
        'json': skipped_lines_json,
        'ast': skipped_lines_ast,
        'other': skipped_lines_other
    }
    print(f"-> Finished {filename}. Found {len(entity_types)} types. Skipped: J={skipped_counts['json']}, A={skipped_counts['ast']}, O={skipped_counts['other']}")

    # Shortened processing report (omits detailed examples)
    file_processing_report_short = {
        'processed_lines': processed_lines,
        'skipped_line_counts': skipped_counts
    }
    return entity_types, file_processing_report_short

def get_dataset_name(filename):
    """Extracts the base dataset name from the filename."""
    # (Implementation remains the same as before)
    match = re.match(r'^([a-zA-Z0-9_.-]+?)\.(?:train|test|dev)', filename)
    if match:
        return match.group(1)
    parts = filename.split('.')
    if len(parts) > 1:
        return parts[0]
    return filename

def calculate_overall_comparison_short(guidex_types, gold_types, set_label):
    """Calculates SHORTENED overlap metrics between GUIDEX and a gold set."""
    if not gold_types:
        return {
            "gold_unique_types_count": 0,
            "status": f"No entity types found in {set_label} set."
        }

    overlapping_types_count = len(guidex_types.intersection(gold_types))
    non_overlapping_gold_types_count = len(gold_types.difference(guidex_types))
    gold_unique_types_count = len(gold_types)

    gold_coverage_percentage = 0.0
    if gold_unique_types_count > 0:
        gold_coverage_percentage = (overlapping_types_count / gold_unique_types_count) * 100

    return {
        "gold_unique_types_count": gold_unique_types_count,
        "overlapping_types_count": overlapping_types_count,
        "gold_coverage_percentage": round(gold_coverage_percentage, 2),
        "gold_types_not_in_guidex_count": non_overlapping_gold_types_count
    }


def process_gold_files_short(file_list, guidex_types, set_label):
    """Processes gold files, groups by dataset, calculates SHORTENED metrics."""
    analysis_results = {
        "files_processed_count": len(file_list),
        "aggregate_unique_types_count": 0,
        "datasets": defaultdict(lambda: {
            "files_count": 0,
            "aggregate_unique_types_count": 0,
            "average_overlap_percentage": 0.0,
            "per_file_reports": {} # Will contain only counts/percentages
        })
    }
    all_set_types = set()

    print(f"\n--- Processing {len(file_list)} {set_label.upper()} files ---")

    for gold_file in file_list:
        filename = os.path.basename(gold_file)
        dataset_name = get_dataset_name(filename)
        if not dataset_name:
            print(f"Warning: Could not determine dataset name for {filename}. Skipping.")
            continue

        file_types, file_report_short = extract_entity_types_from_file(gold_file)
        all_set_types.update(file_types)

        dataset_group = analysis_results["datasets"][dataset_name]
        # dataset_group["files"].append(filename) # Omit list of files for shorter report
        dataset_group["files_count"] += 1
        # Don't store aggregate list per dataset: dataset_group["aggregate_unique_types_list"].update(file_types)

        overlap_count = len(guidex_types.intersection(file_types))
        total_types_in_file = len(file_types)
        overlap_percentage = round((overlap_count / total_types_in_file * 100), 2) if total_types_in_file > 0 else 0.0

        dataset_group["per_file_reports"][filename] = {
            "unique_types_count": total_types_in_file,
            "overlap_with_guidex": {
                "count": overlap_count,
                "percentage": overlap_percentage
                # Omit "types_list"
            },
            "processing_details": file_report_short # Include shortened processing info
        }

    # Post-process: Calculate averages and finalize dataset group info
    for dataset_name, dataset_group in analysis_results["datasets"].items():
        # Calculate aggregate count for the dataset without storing the list
        dataset_types = set().union(*(types_per_file['overlap_with_guidex']['types_list']
                                     for filename, types_per_file in dataset_group["per_file_reports"].items())) if dataset_group["per_file_reports"] else set()

        all_dataset_types_agg = set()
        for fn in dataset_group["per_file_reports"]:
            # Need to re-extract types per file if not stored, or recalculate from stored data
            # Let's recalculate from the files we stored per_file_reports for.
            # This is inefficient, ideally we would store the set per dataset temporarily
            # Re-thinking: aggregate count needs the set. Let's build it here.
             report_for_file = dataset_group["per_file_reports"][fn]
             # To get the types back, we'd need to have stored them or re-read.
             # Let's just count the unique types found across the files processed for this dataset.
             # We can get the individual counts, but not the aggregate without the sets.
             # OK, let's store the set temporarily per dataset during processing.

             # ---- Correction: Need to aggregate types per dataset efficiently ----
             pass # See adjusted logic below


    # ---- Adjusted Aggregation Logic ----
    temp_dataset_types = defaultdict(set)
    for filename, report_dict in analysis_results["datasets"].items(): # Iterate through datasets
         for file_basename, file_report_entry in report_dict["per_file_reports"].items():
              # Re-extract types temporarily ONLY for aggregation - this is still inefficient
              # A better way is to store the sets temporarily during the main loop
              # Let's modify the main loop slightly
              pass # Logic moved to main loop

    # Re-run aggregation after main loop using stored sets
    for dataset_name, dataset_group in analysis_results["datasets"].items():
        # The set `temp_dataset_types[dataset_name]` now holds all unique types for this dataset
        dataset_group["aggregate_unique_types_count"] = len(temp_dataset_types[dataset_name])
        # Omit "aggregate_unique_types_list"

        overlap_percentages = []
        for file_report in dataset_group["per_file_reports"].values():
            if file_report["unique_types_count"] > 0:
                 overlap_percentages.append(file_report["overlap_with_guidex"]["percentage"])

        avg_overlap = np.mean(overlap_percentages) if overlap_percentages else 0.0
        dataset_group["average_overlap_percentage"] = round(float(np.nan_to_num(avg_overlap)), 2)
        # Remove the temporary set now calculation is done
        # del temp_dataset_types[dataset_name] # Clean up if needed, though not strictly necessary


    analysis_results["aggregate_unique_types_count"] = len(all_set_types)
    # Omit "aggregate_unique_types_list"

    analysis_results["datasets"] = dict(analysis_results["datasets"])
    return analysis_results, all_set_types


# --- Main Script (Short Report Version) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate SHORT entity type overlap (GUIDEX vs Gold Train/Test) and generate JSON report.")
    parser.add_argument("--guidex_data", type=str, required=True, help="Path to the GUIDEX .jsonl file.")
    parser.add_argument("--gold_dir", type=str, required=True, help="Directory containing the gold standard files.")
    parser.add_argument("--output_file", type=str, default="overlap_report_short.json", help="Path to save the SHORT JSON report.")
    args = parser.parse_args()

    report = {
        "parameters": {
            "guidex_data_path": args.guidex_data,
            "gold_data_directory": args.gold_dir,
            "output_report_file": args.output_file
        },
        "guidex_processing": {},
        "train_set_analysis": {},
        "test_set_analysis": {},
        "overall_guidex_comparison": {"train": {}, "test": {}}
    }

    # --- Step 1: Process GUIDEX Data ---
    print("--- Extracting entity types from GUIDEX dataset ---")
    guidex_entity_types, guidex_report_details_short = extract_entity_types_from_file(args.guidex_data)
    report["guidex_processing"] = {
        "unique_types_count": len(guidex_entity_types),
        # Omit "unique_types_list"
        "processing_details": guidex_report_details_short # Short version
    }
    if 'error' in guidex_report_details_short:
        print(f"Fatal Error processing GUIDEX file: {guidex_report_details_short['error']}. Aborting.")
        with open(args.output_file, 'w', encoding='utf-8') as f: json.dump(report, f, indent=4)
        print(f"Partial error report saved to {args.output_file}")
        exit()
    elif not guidex_entity_types:
         print("Warning: No entity types extracted from GUIDEX data.")

    # --- Step 2 & 3 Combined Loop for Gold Data ---
    all_train_types = set()
    all_test_types = set()
    temp_dataset_types_train = defaultdict(set) # Temporary store for aggregation
    temp_dataset_types_test = defaultdict(set) # Temporary store for aggregation

    report["train_set_analysis"] = {
        "files_processed_count": 0, "aggregate_unique_types_count": 0, "datasets": defaultdict(lambda: {
            "files_count": 0, "aggregate_unique_types_count": 0, "average_overlap_percentage": 0.0, "per_file_reports": {} })
    }
    report["test_set_analysis"] = {
        "files_processed_count": 0, "aggregate_unique_types_count": 0, "datasets": defaultdict(lambda: {
            "files_count": 0, "aggregate_unique_types_count": 0, "average_overlap_percentage": 0.0, "per_file_reports": {} })
    }

    # Process TRAIN files
    train_files = sorted(glob.glob(os.path.join(args.gold_dir, '*.train.*.jsonl')))
    if train_files:
        print(f"\n--- Processing {len(train_files)} TRAIN files ---")
        report["train_set_analysis"]["files_processed_count"] = len(train_files)
        for gold_file in train_files:
            filename = os.path.basename(gold_file)
            dataset_name = get_dataset_name(filename)
            if not dataset_name: continue

            file_types, file_report_short = extract_entity_types_from_file(gold_file)
            all_train_types.update(file_types)
            temp_dataset_types_train[dataset_name].update(file_types) # Aggregate types per dataset

            dataset_group = report["train_set_analysis"]["datasets"][dataset_name]
            dataset_group["files_count"] += 1

            overlap_count = len(guidex_entity_types.intersection(file_types))
            total_types_in_file = len(file_types)
            overlap_percentage = round((overlap_count / total_types_in_file * 100), 2) if total_types_in_file > 0 else 0.0

            dataset_group["per_file_reports"][filename] = {
                "unique_types_count": total_types_in_file,
                "overlap_with_guidex": {"count": overlap_count, "percentage": overlap_percentage },
                "processing_details": file_report_short
            }
        report["train_set_analysis"]["aggregate_unique_types_count"] = len(all_train_types)

        # Calculate averages for TRAIN
        for dataset_name, dataset_group in report["train_set_analysis"]["datasets"].items():
             dataset_group["aggregate_unique_types_count"] = len(temp_dataset_types_train[dataset_name])
             overlap_percentages = [fr["overlap_with_guidex"]["percentage"] for fr in dataset_group["per_file_reports"].values() if fr["unique_types_count"] > 0]
             avg_overlap = np.mean(overlap_percentages) if overlap_percentages else 0.0
             dataset_group["average_overlap_percentage"] = round(float(np.nan_to_num(avg_overlap)), 2)
        report["train_set_analysis"]["datasets"] = dict(report["train_set_analysis"]["datasets"]) # Convert defaultdict

    else:
         print("\n--- No Gold TRAIN files found ---")
         report["train_set_analysis"]["status"] = "No *.train.*.jsonl files found."


    # Process TEST files
    test_files = sorted(glob.glob(os.path.join(args.gold_dir, '*.test.jsonl')))
    if test_files:
        print(f"\n--- Processing {len(test_files)} TEST files ---")
        report["test_set_analysis"]["files_processed_count"] = len(test_files)
        for gold_file in test_files:
            filename = os.path.basename(gold_file)
            dataset_name = get_dataset_name(filename)
            if not dataset_name: continue

            file_types, file_report_short = extract_entity_types_from_file(gold_file)
            all_test_types.update(file_types)
            temp_dataset_types_test[dataset_name].update(file_types) # Aggregate types per dataset

            dataset_group = report["test_set_analysis"]["datasets"][dataset_name]
            dataset_group["files_count"] += 1

            overlap_count = len(guidex_entity_types.intersection(file_types))
            total_types_in_file = len(file_types)
            overlap_percentage = round((overlap_count / total_types_in_file * 100), 2) if total_types_in_file > 0 else 0.0

            dataset_group["per_file_reports"][filename] = {
                "unique_types_count": total_types_in_file,
                "overlap_with_guidex": {"count": overlap_count, "percentage": overlap_percentage },
                "processing_details": file_report_short
            }
        report["test_set_analysis"]["aggregate_unique_types_count"] = len(all_test_types)

        # Calculate averages for TEST
        for dataset_name, dataset_group in report["test_set_analysis"]["datasets"].items():
             dataset_group["aggregate_unique_types_count"] = len(temp_dataset_types_test[dataset_name])
             overlap_percentages = [fr["overlap_with_guidex"]["percentage"] for fr in dataset_group["per_file_reports"].values() if fr["unique_types_count"] > 0]
             avg_overlap = np.mean(overlap_percentages) if overlap_percentages else 0.0
             dataset_group["average_overlap_percentage"] = round(float(np.nan_to_num(avg_overlap)), 2)
        report["test_set_analysis"]["datasets"] = dict(report["test_set_analysis"]["datasets"]) # Convert defaultdict

    else:
         print("\n--- No Gold TEST files found ---")
         report["test_set_analysis"]["status"] = "No *.test.jsonl files found."


    # --- Step 4: Calculate Overall GUIDEX Comparison (Short Version) ---
    report["overall_guidex_comparison"]["train"] = calculate_overall_comparison_short(
        guidex_entity_types, all_train_types, "train"
    )
    report["overall_guidex_comparison"]["test"] = calculate_overall_comparison_short(
        guidex_entity_types, all_test_types, "test"
    )

    # --- Step 5: Write JSON Report ---
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4)
        print(f"\nAnalysis complete. Short report saved to: {args.output_file}")
    except Exception as e:
        print(f"\nError writing final JSON report to {args.output_file}: {e}")
        

#TO EXECUTE THIS:
# python overlap_report_short.py \
#     --guidex_data /sorgin1/users/neildlf/GoLLIE-dev/data/pretraining_data_processed_w_examples/train_data_10k_valid.jsonl \
#     --gold_dir /sorgin1/users/neildlf/GoLLIE-dev/data/processed_w_examples \
#     --output_file overlap_analysis/overlap_report_short.json