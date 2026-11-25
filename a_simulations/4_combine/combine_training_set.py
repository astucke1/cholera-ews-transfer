####################################################################
# combine_training_set.py -- stack the SIRB and SIARVB outputs and split (pipeline step 4)
#
# two things:
#
#  1 stack -- takes all the per-batch outputs from sirb and siarvb and combines them into
#    one dataset. each sim gets a unique sequence_id across both models. residual files get
#    copied into one folder renamed as resids_{sequence_id}.csv. labels and bifurcation times
#    get merged into master csvs.
#
#  2 split -- randomly assigns each sequence_id to train (70), val (15), or test (15) and
#    saves as groups_all.csv.
#
# Output: output/combined_all_models/
####################################################################

import os
import re
import glob
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split


# ---- path convention ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# each model's output lives inside its own simulation folder
model_paths = {
    "SIRB":   os.path.join(REPO_ROOT, "a_simulations", "1_sirb", "output", "SIRB"),
    "SIARVB": os.path.join(REPO_ROOT, "a_simulations", "2_siarvb", "output", "SIARVB"),
}

# combined output goes in this script's own folder
dest_dir       = os.path.join(SCRIPT_DIR, "output", "combined_all_models")
resids_all_dir = os.path.join(dest_dir, "resids_all")

os.makedirs(dest_dir, exist_ok=True)
os.makedirs(resids_all_dir, exist_ok=True)


# stack

sequence_id_counter = 1
all_labels = []
all_bif_points = []
total_processed = 0

for model_name, model_folder in model_paths.items():
    print(f"\nProcessing model: {model_name}")

    labels_folder = os.path.join(model_folder, "output_labels")
    resids_folder = os.path.join(model_folder, "output_resids")

    label_pattern = os.path.join(labels_folder, "label_batch*.csv")
    label_files = sorted(glob.glob(label_pattern))

    if not label_files:
        print(f"  Warning: no label files found in {labels_folder}. Skipping model.")
        continue

    batch_numbers = []
    for lf in label_files:
        match = re.search(r'label_batch(\d+)\.csv', os.path.basename(lf))
        if match:
            batch_numbers.append(int(match.group(1)))
    batch_numbers.sort()

    print(f"  Found {len(batch_numbers)} batches: {batch_numbers[0]}..{batch_numbers[-1]}")

    for batch_i in batch_numbers:
        label_file = os.path.join(labels_folder, f"label_batch{batch_i}.csv")
        bif_file   = os.path.join(labels_folder, f"bifurcation_points_batch{batch_i}.csv")

        if not os.path.isfile(bif_file):
            print(f"  Warning: {bif_file} not found. Skipping batch {batch_i}")
            continue

        df_label = pd.read_csv(label_file, header=None, names=["class_label"])
        df_bif   = pd.read_csv(bif_file, header=None, names=["bif_point"])

        if len(df_label) != len(df_bif):
            print(f"  Warning: row mismatch in batch {batch_i}. Skipping.")
            continue

        num_sims = len(df_label)

        for row_idx in range(num_sims):
            label_val = df_label.iloc[row_idx]["class_label"]
            bif_val   = df_bif.iloc[row_idx]["bif_point"]
            new_id    = sequence_id_counter

            # copy the residual file with its new sequence_ID name
            sim_idx = row_idx + 1
            resid_name = f"resids_batch{batch_i}_sim{sim_idx}.csv"
            src_path = os.path.join(resids_folder, resid_name)

            if os.path.isfile(src_path):
                dst_path = os.path.join(resids_all_dir, f"resids_{new_id}.csv")
                shutil.copy2(src_path, dst_path)
            else:
                print(f"   Warning: Missing {src_path}. Continuing anyway.")

            all_labels.append({
                "sequence_ID": new_id,
                "class_label": label_val,
                "model_name": model_name,
                "batch": batch_i
            })

            all_bif_points.append({
                "sequence_ID": new_id,
                "t_transition": bif_val
            })

            sequence_id_counter += 1
            total_processed += 1

df_labels_all = pd.DataFrame(all_labels)
df_bif_all    = pd.DataFrame(all_bif_points)

if df_labels_all.empty:
    print("\nNo labels found. Something went wrong with the file paths.")
else:
    df_labels_all.set_index("sequence_ID", inplace=True)
    labels_csv = os.path.join(dest_dir, "labels_all.csv")
    df_labels_all.to_csv(labels_csv)
    print(f"\nCombined labels saved to: {labels_csv}")
    print(f"Total sequences: {total_processed}")

if df_bif_all.empty:
    print("\nNo bifurcation points found.")
else:
    df_bif_all.set_index("sequence_ID", inplace=True)
    bif_csv = os.path.join(dest_dir, "bifurcation_points_all.csv")
    df_bif_all.to_csv(bif_csv)
    print(f"Bifurcation points saved to: {bif_csv}")

print(f"Residual files copied to: {resids_all_dir}")


# t/v/t split

if not df_labels_all.empty:
    print("\nSplitting into train/val/test ...")

    train_ids, temp_ids = train_test_split(
        df_labels_all.index, test_size=0.3, random_state=42
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=42
    )

    # dataset_ID: 1=train, 2=validation, 3=test
    rows = []
    for sid in df_labels_all.index:
        if sid in train_ids:
            ds_id = 1
        elif sid in val_ids:
            ds_id = 2
        else:
            ds_id = 3
        rows.append({"sequence_ID": sid, "dataset_ID": ds_id})

    df_groups = pd.DataFrame(rows).set_index("sequence_ID")
    groups_csv = os.path.join(dest_dir, "groups_all.csv")
    df_groups.to_csv(groups_csv)

    print(f"Groups saved to: {groups_csv}")
    print(f"  Train: {len(train_ids)}")
    print(f"  Validation: {len(val_ids)}")
    print(f"  Test: {len(test_ids)}")

print("\nDone.")
