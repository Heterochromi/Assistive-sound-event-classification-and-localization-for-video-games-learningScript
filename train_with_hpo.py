# %%
from src.loadset import create_dataloaders
from src.dynamic_batching import find_optimal_batch_size
from src.setup_training import setup_training
from src.run_epoch_and_eval import train_one_epoch , evaluate , find_optimal_threshold
import torch
from IPython.display import clear_output

# %%
import time

import matplotlib.pyplot as plt

from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

from pyre_extensions import assert_is_instance

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
HPO_client = Client()

# %%
# Configure and experiment with the desired parameters
parameters=[
        # Choose embedding_dim from values divisible by 2..6 to ensure divisibility for any sampled num_heads
        ChoiceParameterConfig(
            name="embedding_dim",
            values=[120, 180, 240, 300, 360, 420, 480],
            parameter_type="int",
        ),
        RangeParameterConfig(
            name="learning_rate",
            bounds=(1e-5, 1e-2),
            parameter_type="float",
            scaling="log",
        ),
        RangeParameterConfig(
            name="n_conv_layers",
            bounds=(2, 4),
            parameter_type="int",
            scaling="linear",
        ),
        RangeParameterConfig(
            name="num_layers",
            bounds=(2, 8),
            parameter_type="int",
            scaling="linear",
        ),
        RangeParameterConfig(
            name="num_heads",
            bounds=(2, 6),
            parameter_type="int",
            scaling="linear",
        ),
        RangeParameterConfig(
            name="mlp_ratio",
            bounds=(1.0, 4.0),
            parameter_type="float",
            scaling="linear",
        ),
        RangeParameterConfig(
            name="dropout_rate",
            bounds=(0.1, 0.5),
            parameter_type="float",
            scaling="linear",
        ),
        RangeParameterConfig(
            name="attention_dropout",
            bounds=(0.0, 0.5),
            parameter_type="float",
            scaling="linear",
        )
    ]

# %%
HPO_client.configure_experiment(parameters=parameters)

# %%
HPO_client.configure_optimization(
    objective="exact_match_accuracy",
    outcome_constraints=[
        "element_wise_accuracy >= 0.85",
        "f1_micro >= 0.85",
        "f1_macro >= 0.70",
    ],
)

# %%
from src.model_cct import MultiLabelCCT

# %%
def load_model_with_hpo_parameters(hpo_parameters):
    model = MultiLabelCCT(
        img_size=(192, 668),
        embedding_dim=hpo_parameters['embedding_dim'],
        n_input_channels=3,
        n_conv_layers=hpo_parameters['n_conv_layers'],
        kernel_size=7,
        stride=2,
        padding=3,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        num_layers=hpo_parameters['num_layers'],
        num_heads=hpo_parameters['num_heads'],
        mlp_ratio=hpo_parameters['mlp_ratio'],
        dropout_rate=hpo_parameters['dropout_rate'],
        attention_dropout=hpo_parameters['attention_dropout'],
        num_classes=15,
        positional_embedding='learnable',
    )
    model.to(device)
    return model

# %%
epoch = 25

# %%
CSV_PATH = "/home/baraa/Desktop/testroch/DSED_strong_label/dataset/metadata/eval/combined_weak_labels.csv"
IMG_DIR = '/home/baraa/Desktop/testroch/DSED_strong_label/dataset/audio/eval/combined_mel'


# %%
def get_pos_weights(train_loader):
    all_labels = []
    for _, labels in train_loader.dataset:
        all_labels.append(labels)
    all_labels = torch.stack(all_labels)
    
    # Calculate positive class frequencies
    pos_counts = all_labels.sum(dim=0)
    neg_counts = len(all_labels) - pos_counts
    
    # Use a more conservative weighting scheme
    # pos_weights = torch.sqrt(neg_counts / (pos_counts + 1e-8))
    pos_weights = neg_counts / (pos_counts + 1e-8)
    pos_weights = torch.clamp(pos_weights, min=1.0, max=15.0)
    return pos_weights

# %%
# history = {
#     'train_loss': [],
#     'val_loss': [],
#     'exact_match_accuracy': [],
#     'hamming_loss': [],
#     'f1_macro': [],
#     'f1_micro': [],
#     'jaccard_index': [],
#     'element_wise_accuracy': [],
#     'positive_element_wise_accuracy': [],
# }

# %%
for _ in range(15): # Run 15 rounds of 1 trial each
    trials = HPO_client.get_next_trials(max_trials=1)
    torch.cuda.empty_cache()
    for trial_i , parameters in trials.items():
        # training set up
        model = load_model_with_hpo_parameters(parameters)
        input_shape = (3, 192, 668) 
        initial_batch_size = 500 # A reasonable starting point
        optimal_batch_size = find_optimal_batch_size(model, input_shape, initial_batch_size, device , memory_usage_fraction=0.9)
        train_loader, val_loader, mlb = create_dataloaders(CSV_PATH, IMG_DIR , batch_size=optimal_batch_size, val_split=0.20 , height=192 , width=668)
        pos_weights = get_pos_weights(train_loader)
        pos_weights = pos_weights.to(device)
        criterion, optimizer, scheduler = setup_training(model, learning_rate=parameters['learning_rate'], pos_weights=pos_weights , total_steps_scheduler=len(train_loader) * epoch , use_focal_loss = True , T_max=60)
        # end of training set up
        for i in range(0,epoch):
            end_idx = (i + 1) * optimal_batch_size
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            optimal_threshold = find_optimal_threshold(model, val_loader, device)
            metrics = evaluate(model, val_loader, criterion, device , predictions_threshold=optimal_threshold)
            raw_data = {
                "exact_match_accuracy": metrics['exact_match_accuracy'],
                "element_wise_accuracy": metrics['element_wise_accuracy'],
                "f1_macro": metrics['f1_macro'],
                "f1_micro": metrics['f1_micro'],
                "jaccard_index": metrics['jaccard_index'],
                "positive_element_wise_accuracy": metrics['positive_element_wise_accuracy'],
            }
            #--- Update history ---
            # history['train_loss'].append(train_loss)
            # history['val_loss'].append(metrics['loss'])
            # history['exact_match_accuracy'].append(metrics['exact_match_accuracy'])
            # history['hamming_loss'].append(metrics['hamming_loss'])
            # history['f1_macro'].append(metrics['f1_macro'])
            # history['f1_micro'].append(metrics['f1_micro'])
            # history['jaccard_index'].append(metrics['jaccard_index'])
            # history['element_wise_accuracy'].append(metrics['element_wise_accuracy'])
            # history['positive_element_wise_accuracy'].append(metrics['positive_element_wise_accuracy'])
            clear_output(wait=True)
            print(f"Trial {trial_i} - Epoch {i + 1} - Train Loss: {train_loss:.4f} - Val Loss: {metrics['loss']:.4f} - Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f} - Hamming Loss: {metrics['hamming_loss']:.4f} - F1 Macro: {metrics['f1_macro']:.4f} - F1 Micro: {metrics['f1_micro']:.4f} - Jaccard Index: {metrics['jaccard_index']:.4f} - Element-wise Accuracy: {metrics['element_wise_accuracy']:.4f} - Positive Element-wise Accuracy: {metrics['positive_element_wise_accuracy']:.4f}")
            if i == epoch - 1:
                HPO_client.complete_trial(
                    trial_index=trial_i,
                    raw_data=raw_data,
                    progression=end_idx,  # Use the index of the last example in the batch as the progression value
                )
                break
            HPO_client.attach_data(
                trial_index=trial_i,
                raw_data=raw_data,
                progression=end_idx,
            )
            if HPO_client.should_stop_trial_early(trial_index=trial_i):
                HPO_client.mark_trial_early_stopped(trial_index=trial_i)
                break

#ONLY USE get_pareto_frontier if you are doing multiple objective trials , single objective trials can use get_best_parameterization
# %%
# frontier = HPO_client.get_pareto_frontier()

# # Frontier is a list of tuples, where each tuple contains the parameters, the metric readings, the trial index, and the arm name for a point on the Pareto frontier
# for parameters, metrics, trial_index, arm_name in frontier:
#     print(f"Trial {trial_index} with {parameters=} and {metrics=}\n")

# %%
# cards = HPO_client.compute_analyses(display=True)



# %%
best_parameters, prediction, index, name = HPO_client.get_best_parameterization()
print("Best Parameters:", best_parameters)
print("Prediction (mean, variance):", prediction)

# Save best parameters to disk
import json, os, datetime

save_dir = "artifacts"
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(save_dir, f"best_hpo_params_{timestamp}.json")

payload = {
    "best_parameters": best_parameters,
    "prediction_mean": float(prediction[0]),
    "prediction_variance": float(prediction[1]),
    "trial_index": int(index),
    "arm_name": str(name),
}

with open(save_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(f"Saved best parameters to {save_path}")



