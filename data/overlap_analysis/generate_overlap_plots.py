import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os
import numpy as np
import textwrap # For wrapping long labels

def categorize_dataset(dataset_name):
    """Categorizes dataset names into broader task types."""
    name_lower = dataset_name.lower()
    if 'ner' in name_lower:
        return 'NER'
    elif 'eae' in name_lower or '.ee' in name_lower: # Check for Event Extraction patterns
        return 'EAE/EE'
    elif 're' in name_lower or 'rc' in name_lower or 'sf' in name_lower: # Relation/Slot Filling
        return 'RE/RC/SF'
    elif 'ver' in name_lower: # Verification might be separate
         return 'Verification'
    # Add more specific rules if needed
    elif 'casie' in name_lower: # CASIE is often considered Event-related
         return 'EAE/EE'
    elif 'rams' in name_lower: # RAMS is Event Argument Structure
         return 'EAE/EE'
    elif 'wikievents' in name_lower: # WikiEvents spans multiple types but often event-centric
         return 'EAE/EE' # Group with EAE/EE for simplicity
    else:
        return 'Other'

def plot_aggregate_summary(report_data, output_dir):
    """Generates Plot 1: Aggregate Summary Statistics."""
    print("Generating Plot 1: Aggregate Summary...")
    guidex_count = report_data.get("guidex_processing", {}).get("unique_types_count", 0)
    train_analysis = report_data.get("train_set_analysis", {})
    test_analysis = report_data.get("test_set_analysis", {})
    overall_comp = report_data.get("overall_guidex_comparison", {})

    train_count = train_analysis.get("aggregate_unique_types_count", 0)
    test_count = test_analysis.get("aggregate_unique_types_count", 0)

    train_overlap_perc = overall_comp.get("train", {}).get("gold_coverage_percentage", 0.0)
    test_overlap_perc = overall_comp.get("test", {}).get("gold_coverage_percentage", 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Aggregate Entity Type Summary', fontsize=16)

    # Subplot 1: Unique Type Counts
    sns.set_style("whitegrid")
    counts_data = {
        'Set': ['GUIDEX', 'Gold Train', 'Gold Test'],
        'Count': [guidex_count, train_count, test_count]
    }
    sns.barplot(x='Set', y='Count', data=counts_data, ax=axes[0], palette="viridis")
    axes[0].set_title('Total Unique Entity Types')
    axes[0].set_ylabel('Number of Unique Types')
    axes[0].ticklabel_format(style='plain', axis='y') # Prevent scientific notation
    # Add count labels on bars
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='{:,.0f}')


    # Subplot 2: Overall Overlap Percentage
    overlap_data = {
        'Set': ['Train Set', 'Test Set'],
        'Overlap (%)': [train_overlap_perc, test_overlap_perc]
    }
    sns.barplot(x='Set', y='Overlap (%)', data=overlap_data, ax=axes[1], palette="plasma")
    axes[1].set_title('GUIDEX Coverage of Gold Set Types')
    axes[1].set_ylabel('Overlap Percentage (%)')
    axes[1].set_ylim(0, 100)
    # Add percentage labels on bars
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.2f%%')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    save_path = os.path.join(output_dir, '1_aggregate_summary.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_per_dataset_overlap(report_data, set_key, output_dir):
    """Generates Plot 2: Per-Dataset Average Overlap (Train or Test)."""
    set_label = set_key.replace("_set_analysis", "").capitalize()
    print(f"Generating Plot 2: Per-Dataset Overlap ({set_label})...")
    analysis_data = report_data.get(set_key, {}).get("datasets", {})
    if not analysis_data:
        print(f"No dataset analysis found for {set_label} set. Skipping plot.")
        return

    dataset_overlaps = []
    for name, data in analysis_data.items():
        # Wrap long names for better plotting
        wrapped_name = '\n'.join(textwrap.wrap(name, width=30)) # Adjust width as needed
        dataset_overlaps.append({
            'Dataset': wrapped_name, # Use wrapped name for plot label
            'Original Name': name, # Keep original for sorting if needed
            'Average Overlap (%)': data.get('average_overlap_percentage', 0.0)
        })

    if not dataset_overlaps:
        print(f"No dataset overlap data to plot for {set_label} set.")
        return

    df = pd.DataFrame(dataset_overlaps)
    df = df.sort_values(by='Average Overlap (%)', ascending=False)

    # Dynamic Figure Size based on number of datasets
    num_datasets = len(df)
    height_per_dataset = 0.3 # Inches per dataset bar
    min_height = 6
    fig_height = max(min_height, num_datasets * height_per_dataset)

    plt.figure(figsize=(10, fig_height))
    sns.set_style("whitegrid")
    barplot = sns.barplot(x='Average Overlap (%)', y='Dataset', data=df, orient='h', palette='coolwarm')
    plt.title(f'Average GUIDEX Entity Type Overlap per Dataset ({set_label} Set)')
    plt.xlabel('Average Overlap Percentage (%)')
    plt.ylabel('Dataset')
    plt.xlim(0, 100)

    # Add percentage labels to the right of the bars
    for index, value in enumerate(df['Average Overlap (%)']):
        plt.text(value + 1, # Position slightly right of bar end
                 index, # Y position
                 f'{value:.2f}%',
                 color='black',
                 va='center',
                 fontsize=8) # Adjust fontsize as needed

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'2_per_dataset_overlap_{set_label.lower()}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_task_type_overlap(report_data, output_dir):
    """Generates Plot 4 (renamed): Task-Type Overlap Comparison."""
    print("Generating Plot 3: Task-Type Overlap Comparison...")
    train_analysis = report_data.get("train_set_analysis", {}).get("datasets", {})
    test_analysis = report_data.get("test_set_analysis", {}).get("datasets", {})

    if not train_analysis and not test_analysis:
        print("No train or test dataset analysis found. Skipping task type plot.")
        return

    task_overlaps = []

    # Process Train Data
    for name, data in train_analysis.items():
        task_type = categorize_dataset(name)
        task_overlaps.append({
            'Task Type': task_type,
            'Set': 'Train',
            'Average Overlap (%)': data.get('average_overlap_percentage', 0.0),
            'Dataset Count': 1 # Count datasets per type
        })

    # Process Test Data
    for name, data in test_analysis.items():
        task_type = categorize_dataset(name)
        task_overlaps.append({
            'Task Type': task_type,
            'Set': 'Test',
            'Average Overlap (%)': data.get('average_overlap_percentage', 0.0),
            'Dataset Count': 1
        })

    if not task_overlaps:
        print("Could not gather any task type overlap data. Skipping plot.")
        return

    df = pd.DataFrame(task_overlaps)

    # Calculate the average overlap *across datasets* for each task type and set
    # Also count the number of datasets per task type
    agg_df = df.groupby(['Task Type', 'Set']).agg(
        MeanOverlap=('Average Overlap (%)', 'mean'),
        DatasetCount=('Dataset Count', 'sum') # Sum the 1s
    ).reset_index()

    # Sort Task Types for consistent plotting order (e.g., alphabetical or custom)
    # Custom order might be: NER, EAE/EE, RE/RC/SF, Verification, Other
    custom_order = ['NER', 'EAE/EE', 'RE/RC/SF', 'Verification', 'Other']
    # Filter agg_df to only include task types present and order them
    ordered_types = [t for t in custom_order if t in agg_df['Task Type'].unique()]
    agg_df['Task Type'] = pd.Categorical(agg_df['Task Type'], categories=ordered_types, ordered=True)
    agg_df = agg_df.sort_values('Task Type')


    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    barplot = sns.barplot(x='Task Type', y='MeanOverlap', hue='Set', data=agg_df, palette='muted')

    plt.title('Average GUIDEX Entity Type Overlap by Task Type')
    plt.xlabel('Inferred Task Type')
    plt.ylabel('Average Overlap Percentage (%)')
    plt.ylim(0, 100)
    plt.legend(title='Gold Set')
    plt.xticks(rotation=0) # Keep labels horizontal if few categories

    # Add labels (mean percentage and dataset count) on bars
    for container in barplot.containers:
        labels = [f'{v.get_height():.2f}%\n(n={agg_df.iloc[v.get_x() + v.get_width()/2., "DatasetCount"]:.0f})'
                  if v.get_height() > 0 else ''
                  for v in container]

        labels_with_counts = []
        for i, bar in enumerate(container.patches):
            height = bar.get_height()
            if height <= 0:
                 labels_with_counts.append('')
                 continue
             
            set_name = container.get_label() # Get hue label ('Train' or 'Test')
            task_type = barplot.get_xticklabels()[int(bar.get_x() + bar.get_width() / 2)].get_text() # Get category label

            count_row = agg_df[(agg_df['Task Type'] == task_type) & (agg_df['Set'] == set_name)]
            count = count_row['DatasetCount'].iloc[0] if not count_row.empty else '?'

            labels_with_counts.append(f'{height:.1f}%\n(n={count})')


        barplot.bar_label(container, labels=labels_with_counts, label_type='edge', fontsize=8, padding=2)


    plt.tight_layout()
    save_path = os.path.join(output_dir, '3_task_type_overlap.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# --- Main Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots visualizing entity type overlap from overlap_report_short.json.")
    parser.add_argument("report_file", help="Path to the input overlap_report_short.json file.")
    parser.add_argument("--output_dir", default="overlap_plots", help="Directory to save the generated plots.")
    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.report_file):
        print(f"Error: Report file not found at {args.report_file}")
        exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the report data
    try:
        with open(args.report_file, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from {args.report_file}: {e}")
        exit(1)
    except Exception as e:
        print(f"Error loading report file {args.report_file}: {e}")
        exit(1)

    # Generate Plots
    plot_aggregate_summary(report_data, args.output_dir)
    plot_per_dataset_overlap(report_data, "train_set_analysis", args.output_dir)
    plot_per_dataset_overlap(report_data, "test_set_analysis", args.output_dir)
    plot_task_type_overlap(report_data, args.output_dir)

    print("\nPlot generation complete.")
    print(f"Plots saved in: {args.output_dir}")