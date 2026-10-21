"""
Utility functions used by all clients and for parameter conversion.
"""
import os 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader


# Define the feature and label columns centrally so all clients use the same order
FEATURE_COLUMNS = [
"Microwave [kW]",
"Fridge [kW]",
"Dishwasher [kW]",
"WasherDrier [kW]",
"Lighting [kW]",
"Furnace [kW]",
"TotalEnergy [kW]",
]
LABEL_COLUMN = "anomaly"




def load_and_preprocess(csv_path, test_size=0.2, random_state=42, batch_size=32):
	# 
	# Load CSV, validate columns, split into train/test, scale locally, and return PyTorch DataLoaders
	# plus counts. Clients scale their own data locally to avoid leaking global stats.

	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV not found: {csv_path}")


	df = pd.read_csv(csv_path)


	# Ensure required columns exist
	for col in FEATURE_COLUMNS + [LABEL_COLUMN]:
		if col not in df.columns:
			raise ValueError(f"Missing column {col} in {csv_path}")


	# Extract features and label
	X = df[FEATURE_COLUMNS].astype(float).values
	y = df[LABEL_COLUMN].astype(int).values


	# Stratify split when possible to keep class balance in train/test
	stratify = y if len(np.unique(y)) > 1 else None
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state, stratify=stratify
	)


	# Local standardization (important in FL)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)


	# Convert to PyTorch tensors
	X_train_t = torch.from_numpy(X_train).float()
	y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)
	X_test_t = torch.from_numpy(X_test).float()
	y_test_t = torch.from_numpy(y_test).float().unsqueeze(1)

	# Create datasets and dataloaders
	train_ds = TensorDataset(X_train_t, y_train_t)
	test_ds = TensorDataset(X_test_t, y_test_t)

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

	# Return loaders, counts and the local scaler
	return train_loader, test_loader, len(y_train), len(y_test), scaler

def get_numpy_parameters(model):
    """Return model parameters as list of NumPy arrays (ordered same as state_dict)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters_from_numpy(model, numpy_parameters):
    """Set model.parameters from a list of NumPy arrays (matching state_dict order)."""
    state_dict = model.state_dict()
    if len(numpy_parameters) != len(state_dict):
        raise ValueError("Mismatch between provided parameters and model state_dict")
    new_state = {}
    for (key, _), arr in zip(state_dict.items(), numpy_parameters):
        new_state[key] = torch.tensor(arr)
    model.load_state_dict(new_state)