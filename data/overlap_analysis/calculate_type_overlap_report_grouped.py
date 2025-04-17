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
    Returns a tuple: (set_of_entity_types, processing_report_dict)
    """
    entity_types = set()
    filename = os.path.basename(filepath)
    print(f"Processing: {filename} for entity types")
    processed_lines = 0
    skipped_lines_json = 0
    skipped_lines_ast = 0
    skipped_lines_other = 0
    skipped_line_details = defaultdict(list)

    try:
        # Pre-count lines for accurate tqdm total
        try:
            with open(filepath, 'r', encoding='utf-8') as f_count:
                num_lines = sum(1 for _ in f_count)
        except Exception:
            num_lines = None # Fallback

        with open(filepath, 'r', encoding='utf-8') as f:
            pbar = tqdm(f, total=num_lines, desc=f"Reading {filename.split('.')[0]}", leave=False)
            for line_num, line in enumerate(pbar, 1):
                processed_lines += 1
                line_content_preview = line[:150].strip() + "..." if len(line) > 150 else line.strip()
                try:
                    data = json.loads(line)
                    labels_str = data.get('labels')

                    if labels_str and isinstance(labels_str, str) and labels_str.strip() and labels_str.strip() != '[]':
                        try:
                            tree = ast.parse(labels_str.strip(), mode='eval')
                            if isinstance(tree, ast.Expression):
                                entity_types.update(extract_class_names_from_ast(tree.body))
                        except SyntaxError as e:
                            skipped_lines_ast += 1
                            if len(skipped_line_details['ast_syntax_error']) < 5: # Limit examples
                                skipped_line_details['ast_syntax_error'].append({'line': line_num, 'error': str(e), 'content': line_content_preview})
                        except Exception as e:
                            skipped_lines_ast += 1
                            if len(skipped_line_details['ast_processing_error']) < 5:
                                skipped_line_details['ast_processing_error'].append({'line': line_num, 'error': str(e), 'content': line_content_preview})
                    elif labels_str and not isinstance(labels_str, str):
                         skipped_lines_other += 1
                         if len(skipped_line_details['non_string_label']) < 5:
                             skipped_line_details['non_string_label'].append({'line': line_num, 'type': str(type(labels_str)), 'content': line_content_preview})

                except json.JSONDecodeError as e:
                    skipped_lines_json += 1
                    if len(skipped_line_details['json_decode_error']) < 5:
                        skipped_line_details['json_decode_error'].append({'line': line_num, 'error': str(e), 'content': line_content_preview})
                except Exception as e:
                     skipped_lines_other += 1
                     if len(skipped_line_details['other_error']) < 5:
                        skipped_line_details['other_error'].append({'line': line_num, 'error': str(e), 'content': line_content_preview})

                pbar.set_postfix({
                    "types": len(entity_types),
                    "skip_json": skipped_lines_json,
                    "skip_ast": skipped_lines_ast,
                    "skip_other": skipped_lines_other
                    }, refresh=False) # Reduce refresh rate slightly

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

    file_processing_report = {
        'processed_lines': processed_lines,
        'skipped_line_counts': skipped_counts,
        'skipped_line_details (examples)': dict(skipped_line_details)
    }
    return entity_types, file_processing_report

def get_dataset_name(filename):
    """Extracts the base dataset name from the filename."""
    # Regex to capture the part before .train, .test, or .dev
    # Handles names like 'ace05', 'conll03', 'crossner.crossner_ai'
    match = re.match(r'^([a-zA-Z0-9_.-]+?)\.(?:train|test|dev)', filename)
    if match:
        return match.group(1)
    # Fallback: if no train/test/dev, maybe it's just the base name? Unlikely given patterns.
    # Or split and try heuristic (less reliable)
    parts = filename.split('.')
    if len(parts) > 1:
        return parts[0] # Simple fallback
    return filename # Cannot determine

def calculate_overall_comparison(guidex_types, gold_types, set_label):
    """Calculates overlap metrics between GUIDEX and a gold set (train or test)."""
    if not gold_types:
        return {
            "gold_unique_types_count": 0,
            "status": f"No entity types found in {set_label} set."
        }

    overlapping_types = guidex_types.intersection(gold_types)
    non_overlapping_gold_types = gold_types.difference(guidex_types)
    non_overlapping_guidex_types = guidex_types.difference(gold_types) # Note: this is relative to *this* gold set

    gold_coverage_percentage = 0.0
    if len(gold_types) > 0:
        gold_coverage_percentage = (len(overlapping_types) / len(gold_types)) * 100

    return {
        "gold_unique_types_count": len(gold_types),
        "overlapping_types_count": len(overlapping_types),
        "gold_coverage_percentage": round(gold_coverage_percentage, 2),
        "gold_types_not_in_guidex_count": len(non_overlapping_gold_types),
        # "guidex_types_not_in_this_gold_set_count": len(non_overlapping_guidex_types), # Less common metric
        "overlapping_types_list": sorted(list(overlapping_types)),
        "gold_types_not_in_guidex_list": sorted(list(non_overlapping_gold_types)),
        # "guidex_types_not_in_this_gold_set_list": sorted(list(non_overlapping_guidex_types))
    }


def process_gold_files(file_list, guidex_types, set_label):
    """Processes a list of gold files (train or test), groups by dataset, and calculates metrics."""
    analysis_results = {
        "files_processed_count": len(file_list),
        "aggregate_unique_types_count": 0,
        "aggregate_unique_types_list": [],
        "datasets": defaultdict(lambda: {
            "files": [],
            "files_count": 0,
            "aggregate_unique_types_count": 0,
            "aggregate_unique_types_list": set(), # Use set for aggregation
            "average_overlap_percentage": 0.0,
            "per_file_reports": {}
        })
    }
    all_set_types = set() # All types found in this set (train or test)

    print(f"\n--- Processing {len(file_list)} {set_label.upper()} files ---")

    for gold_file in file_list:
        filename = os.path.basename(gold_file)
        dataset_name = get_dataset_name(filename)
        if not dataset_name:
            print(f"Warning: Could not determine dataset name for {filename}. Skipping.")
            continue

        file_types, file_report = extract_entity_types_from_file(gold_file)
        all_set_types.update(file_types)

        # Store per-file info under its dataset group
        dataset_group = analysis_results["datasets"][dataset_name]
        dataset_group["files"].append(filename)
        dataset_group["files_count"] += 1
        dataset_group["aggregate_unique_types_list"].update(file_types) # Aggregate types for this dataset

        # Calculate per-file overlap with GUIDEX
        overlap_count = len(guidex_types.intersection(file_types))
        total_types_in_file = len(file_types)
        overlap_percentage = round((overlap_count / total_types_in_file * 100), 2) if total_types_in_file > 0 else 0.0

        dataset_group["per_file_reports"][filename] = {
            "unique_types_count": total_types_in_file,
            "overlap_with_guidex": {
                "count": overlap_count,
                "percentage": overlap_percentage,
                "types_list": sorted(list(guidex_types.intersection(file_types)))
            },
            "processing_details": file_report # Include skipped line info etc.
        }

    # Post-process: Calculate averages and finalize dataset group info
    for dataset_name, dataset_group in analysis_results["datasets"].items():
        dataset_group["aggregate_unique_types_count"] = len(dataset_group["aggregate_unique_types_list"])
        dataset_group["aggregate_unique_types_list"] = sorted(list(dataset_group["aggregate_unique_types_list"])) # Convert set to sorted list

        # Calculate average overlap percentage for the dataset
        overlap_percentages = []
        for file_report in dataset_group["per_file_reports"].values():
            # Only include files that had types in the average calculation
            if file_report["unique_types_count"] > 0:
                 overlap_percentages.append(file_report["overlap_with_guidex"]["percentage"])

        # Use numpy.mean to handle empty list case correctly (returns nan, convert to 0.0)
        avg_overlap = np.mean(overlap_percentages) if overlap_percentages else 0.0
        dataset_group["average_overlap_percentage"] = round(float(np.nan_to_num(avg_overlap)), 2)


    analysis_results["aggregate_unique_types_count"] = len(all_set_types)
    analysis_results["aggregate_unique_types_list"] = sorted(list(all_set_types))

    # Convert defaultdict back to dict for JSON
    analysis_results["datasets"] = dict(analysis_results["datasets"])

    return analysis_results, all_set_types

# --- Main Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate entity type overlap (GUIDEX vs Gold Train/Test) and generate JSON report.")
    parser.add_argument("--guidex_data", type=str, required=True, help="Path to the GUIDEX .jsonl file.")
    parser.add_argument("--gold_dir", type=str, required=True, help="Directory containing the gold standard files.")
    parser.add_argument("--output_file", type=str, default="overlap_report.json", help="Path to save the JSON report.")
    args = parser.parse_args()

    # Initialize the report structure
    report = {
        "parameters": {
            "guidex_data_path": args.guidex_data,
            "gold_data_directory": args.gold_dir,
            "output_report_file": args.output_file
        },
        "guidex_processing": {},
        "train_set_analysis": {}, # To be filled by process_gold_files
        "test_set_analysis": {},  # To be filled by process_gold_files
        "overall_guidex_comparison": { # To be filled after processing both sets
             "train": {},
             "test": {}
        }
    }

    # --- Step 1: Process GUIDEX Data ---
    print("--- Extracting entity types from GUIDEX dataset ---")
    guidex_entity_types, guidex_report_details = extract_entity_types_from_file(args.guidex_data)
    report["guidex_processing"] = {
        "unique_types_count": len(guidex_entity_types),
        "unique_types_list": sorted(list(guidex_entity_types)),
        "processing_details": guidex_report_details
    }
    if 'error' in guidex_report_details:
        print(f"Fatal Error processing GUIDEX file: {guidex_report_details['error']}. Aborting.")
        with open(args.output_file, 'w', encoding='utf-8') as f: json.dump(report, f, indent=4)
        print(f"Partial error report saved to {args.output_file}")
        exit()
    elif not guidex_entity_types:
         print("Warning: No entity types extracted from GUIDEX data. Overlap calculations will be zero.")


    # --- Step 2: Process Gold Train Data ---
    train_files = sorted(glob.glob(os.path.join(args.gold_dir, '*.train.*.jsonl')))
    if train_files:
        train_analysis_dict, all_train_types = process_gold_files(train_files, guidex_entity_types, "train")
        report["train_set_analysis"] = train_analysis_dict
        # Calculate overall comparison for TRAIN set vs GUIDEX
        report["overall_guidex_comparison"]["train"] = calculate_overall_comparison(
            guidex_entity_types, all_train_types, "train"
        )
    else:
        print("\n--- No Gold TRAIN files found ---")
        report["train_set_analysis"]["status"] = "No *.train.*.jsonl files found."
        report["overall_guidex_comparison"]["train"]["status"] = "No train files to compare."
        all_train_types = set() # Ensure it exists


    # --- Step 3: Process Gold Test Data ---
    test_files = sorted(glob.glob(os.path.join(args.gold_dir, '*.test.jsonl')))
    if test_files:
        test_analysis_dict, all_test_types = process_gold_files(test_files, guidex_entity_types, "test")
        report["test_set_analysis"] = test_analysis_dict
        # Calculate overall comparison for TEST set vs GUIDEX
        report["overall_guidex_comparison"]["test"] = calculate_overall_comparison(
            guidex_entity_types, all_test_types, "test"
        )
    else:
        print("\n--- No Gold TEST files found ---")
        report["test_set_analysis"]["status"] = "No *.test.jsonl files found."
        report["overall_guidex_comparison"]["test"]["status"] = "No test files to compare."
        all_test_types = set() # Ensure it exists


    # --- Step 4: Write JSON Report ---
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4)
        print(f"\nAnalysis complete. Report saved to: {args.output_file}")
    except Exception as e:
        print(f"\nError writing final JSON report to {args.output_file}: {e}")