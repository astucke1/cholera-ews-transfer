####################################################################
# train500.py -- train one CNN-LSTM classifier on the combined SIRB + SIARVB set, window W = 500
#
# architecture from Bury et al. (2021):
#   Conv1D -> Dropout -> MaxPool -> LSTM(50) -> Dropout -> LSTM(10) -> Dropout -> Dense(2, softmax)
#
#
#  command line:
#   python train500.py <model_type> <kk>
#   e.g.  python train500.py 1 1 ; python train500.py 2 1
#
####################################################################

import os
import sys
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, LSTM, Dropout, MaxPooling1D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    precision_score, recall_score, classification_report
)


# ---- path convention ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# training data lives in the step-4 output folder
base_dir = os.path.join(
    REPO_ROOT, "a_simulations", "4_last_step_training-prep",
    "output", "combined_all_models"
)

if len(sys.argv) < 3:
    print("Usage: python train500.py <model_type> <kk>")
    sys.exit(1)

model_type = int(sys.argv[1])
kk = int(sys.argv[2])

print(f"Train a classifier of type {model_type} with index {kk} with 500")

labels_csv    = os.path.join(base_dir, "labels_all.csv")
bif_csv       = os.path.join(base_dir, "bifurcation_points_all.csv")
groups_csv    = os.path.join(base_dir, "groups_all.csv")
resids_folder = os.path.join(base_dir, "resids_all")

# 500 point window instead of 100
ts_len = 500
max_time = 1500

df_labels = pd.read_csv(labels_csv, index_col="sequence_ID")
df_bif    = pd.read_csv(bif_csv, index_col="sequence_ID")
df_groups = pd.read_csv(groups_csv, index_col="sequence_ID")

df_labels.sort_index(inplace=True)
df_bif.sort_index(inplace=True)
df_groups.sort_index(inplace=True)

all_ids = df_labels.index.values
print(f"Total sequences in labels_all: {len(all_ids)}")


existing_ids = []
for sid in all_ids:
    resid_path = os.path.join(resids_folder, f"resids_{sid}.csv")
    if os.path.isfile(resid_path):
        existing_ids.append(sid)
    else:
        print(f"  Warning: missing resids_{sid}.csv, skipping")

df_labels = df_labels.loc[existing_ids]
df_bif    = df_bif.loc[existing_ids]
df_groups = df_groups.loc[existing_ids]

seq_ids = df_labels.index.values
print(f"Sequences after file check: {len(seq_ids)}")

sequences = []
transition_points = []
seq_ids_in_order = []

for sid in seq_ids:
    resid_path = os.path.join(resids_folder, f"resids_{sid}.csv")
    df_resid = pd.read_csv(resid_path)
    df_resid["residuals"] = df_resid["residuals"].fillna(0)

    t_transition = df_bif.loc[sid, "t_transition"]
    values = df_resid["residuals"].values

    if (t_transition < max_time - 1) and (t_transition > ts_len):
        start_idx = int(t_transition - ts_len + 1)
        end_idx   = int(t_transition + 1)
        values_segment = values[start_idx:end_idx]
    elif (t_transition < max_time - 1) and (t_transition < ts_len):
        values_segment = values[0:ts_len]
    else:
        values_segment = values[-ts_len:]

    # safety check — pad or truncate to exactly ts_len
    if len(values_segment) < ts_len:
        padded = np.zeros(ts_len)
        padded[-len(values_segment):] = values_segment
        values_segment = padded
    elif len(values_segment) > ts_len:
        values_segment = values_segment[-ts_len:]

    sequences.append(values_segment)
    transition_points.append(t_transition)
    seq_ids_in_order.append(sid)

sequences = np.array(sequences)
print(f"Slicing complete. shape = {sequences.shape}")


for i in range(len(sequences)):
    seq = sequences[i]
    nonzero_vals = np.abs(seq[seq != 0])
    if len(nonzero_vals) > 0:
        mean_val = np.mean(nonzero_vals)
        seq[seq != 0] = seq[seq != 0] / mean_val
    sequences[i] = seq

# using the assigned split from earlier
# 1 = train, 2 = validation, 3 = test

id_to_row = {sid: idx for idx, sid in enumerate(seq_ids_in_order)}

train_seqs, val_seqs, test_seqs = [], [], []
train_tgts, val_tgts, test_tgts = [], [], []

for sid in df_groups.index:
    ds_id = df_groups.loc[sid, "dataset_ID"]
    row_i = id_to_row[sid]

    if ds_id == 1:
        train_seqs.append(sequences[row_i])
        train_tgts.append(df_labels.loc[sid, "class_label"])
    elif ds_id == 2:
        val_seqs.append(sequences[row_i])
        val_tgts.append(df_labels.loc[sid, "class_label"])
    elif ds_id == 3:
        test_seqs.append(sequences[row_i])
        test_tgts.append(df_labels.loc[sid, "class_label"])

train = np.array(train_seqs).reshape(-1, ts_len, 1)
validation = np.array(val_seqs).reshape(-1, ts_len, 1)
test = np.array(test_seqs).reshape(-1, ts_len, 1)

train_target = np.array(train_tgts)
validation_target = np.array(val_tgts)
test_target = np.array(test_tgts)

print(f"Train: {train.shape}  Val: {validation.shape}  Test: {test.shape}")


# hyperparameters from Bury et al. (2021) exactly

CNN_layers          = 1
LSTM_layers         = 1
pool_size_param     = 2
learning_rate_param = 0.0005
dropout_percent     = 0.10
filters_param       = 50
mem_cells           = 50
mem_cells2          = 10
kernel_size_param   = 12
initializer_param   = 'lecun_normal'
epoch_param         = 1500
batch_param         = 1000

model_name = os.path.join(SCRIPT_DIR, f"final_model_{kk}_{model_type}_length{ts_len}.keras")

model = Sequential()

if CNN_layers == 1:
    model.add(Conv1D(
        filters=filters_param,
        kernel_size=kernel_size_param,
        activation='relu',
        padding='same',
        input_shape=(ts_len, 1),
        kernel_initializer=initializer_param
    ))
elif CNN_layers == 2:
    model.add(Conv1D(
        filters=filters_param,
        kernel_size=kernel_size_param,
        activation='relu',
        padding='same',
        input_shape=(ts_len, 1)
    ))
    model.add(Conv1D(
        filters=2 * filters_param,
        kernel_size=kernel_size_param,
        activation='relu',
        padding='same'
    ))

model.add(Dropout(dropout_percent))
model.add(MaxPooling1D(pool_size=pool_size_param))

if LSTM_layers == 1:
    model.add(LSTM(mem_cells, return_sequences=True, kernel_initializer=initializer_param))
    model.add(Dropout(dropout_percent))
elif LSTM_layers == 2:
    model.add(LSTM(mem_cells, return_sequences=True))
    model.add(LSTM(mem_cells, return_sequences=True))
    model.add(Dropout(dropout_percent))

model.add(LSTM(mem_cells2, kernel_initializer=initializer_param))
model.add(Dropout(dropout_percent))
model.add(Dense(2, activation='softmax', kernel_initializer=initializer_param))

adam = Adam(learning_rate=learning_rate_param)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy', 'sparse_categorical_accuracy']
)

chk = ModelCheckpoint(
    filepath=model_name,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=150, verbose=1)   
history = model.fit(
    train, train_target,
    epochs=epoch_param,
    batch_size=batch_param,
    validation_data=(validation, validation_target),
    callbacks=[chk, es]
)

best_model = load_model(model_name)

test_probs = best_model.predict(test)
test_preds = np.argmax(test_probs, axis=1)

acc = accuracy_score(test_target, test_preds)
f1_macro = f1_score(test_target, test_preds, average='macro')
prec_macro = precision_score(test_target, test_preds, average='macro')
recall_macro = recall_score(test_target, test_preds, average='macro')
cm = confusion_matrix(test_target, test_preds)
report = classification_report(test_target, test_preds, digits=3)

print(f"\nTest Accuracy: {acc:.3f}")
print(report)
print(f"F1 (macro): {f1_macro}")
print(f"Precision (macro): {prec_macro}")
print(f"Recall (macro): {recall_macro}")
print(f"Confusion Matrix:\n{cm}")

actual_train_size = len(train)

f1_name = os.path.join(SCRIPT_DIR, f"training_results_{kk}_{model_type}_500.txt")
f2_name = os.path.join(SCRIPT_DIR, f"training_results_{kk}_{model_type}_500.csv")

with open(f1_name, "w") as f_results, open(f2_name, "w") as f_results2:
    f_results.write(report + "\n")
    f_results.write(f"Test Accuracy: {acc:.4f}\n")
    f_results.write(f"F1 (macro): {f1_macro:.4f}\n")
    f_results.write(f"Precision (macro): {prec_macro:.4f}\n")
    f_results.write(f"Recall (macro): {recall_macro:.4f}\n")
    f_results.write(f"Confusion Matrix:\n{cm}\n")
    f_results.write(
        "Simulation %d macro f1: %f. macro avg precision %f. macro avg recall %f. "
        "Kernel size %d. Filters %d. Batch size %d. Epochs %d. seq_len %d. Train size %d. "
        "Mem_cells %d. Mem_cells2 %d. dropout %f. CNN layers %d, LSTM layers %d, "
        "pool_size_param %d, learning_rate_param %f, initializer %s\n"
        % (kk, f1_macro, prec_macro, recall_macro,
           kernel_size_param, filters_param, batch_param, epoch_param,
           ts_len, actual_train_size, mem_cells, mem_cells2, dropout_percent,
           CNN_layers, LSTM_layers, pool_size_param, learning_rate_param,
           initializer_param)
    )
    f_results2.write(
        "%d, %f, %f, %f, %d, %d, %d, %d, %d, %d, %d, %d, %f, %d, %d, %d, %f, %s\n"
        % (kk, f1_macro, prec_macro, recall_macro,
           kernel_size_param, filters_param, batch_param, epoch_param,
           ts_len, actual_train_size, mem_cells, mem_cells2, dropout_percent,
           CNN_layers, LSTM_layers, pool_size_param, learning_rate_param,
           initializer_param)
    )

print(f"\nDone. Best model saved to {model_name}")
print(f"Metrics saved to {f1_name} and {f2_name}")
