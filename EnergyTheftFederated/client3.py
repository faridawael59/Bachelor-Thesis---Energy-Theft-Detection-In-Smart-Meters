
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_and_preprocess, get_numpy_parameters, set_parameters_from_numpy
from sklearn.metrics import precision_score, recall_score, f1_score


# --------------------- Model ---------------------
class MLP(nn.Module):  #declaring my model's architecture
    def __init__(self, in_features=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), #first input layer z=wx+b, Fully connected (Dense) with 64 neurons, 
            nn.ReLU(), #activation, adds non-linearity, keeps positive values, zeroes negative, has no learnable parameters
            nn.Dropout(0.2),# regularization technique to prevent overfitting, randomly sets 20% of inputs to zero during training
            nn.Linear(64, 32), # second hidden layer with 32 neurons
            nn.ReLU(), #  activation
            nn.Linear(32, 1), # output layer with 1 neuron for binary classification
            nn.Sigmoid(), # activation to output probabilities between 0 and 1
        ) # y^​=σ(W3​⋅ReLU(W2​⋅Dropout(ReLU(W1​x+b1​))+b2​)+b3​)

    def forward(self, x): # how the data flows through the model during forward pass
        return self.net(x)

# --------------------- Client ---------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device, n_train, n_test, client_name):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.n_train = n_train
        self.n_test = n_test
        self.client_name = client_name
        self.criterion = nn.BCELoss()
        self.round_counter = 0
        self.initial_eval_done = False

    def get_parameters(self, config=None):
        return get_numpy_parameters(self.model)

    def fit(self, parameters, config):# receives global model parameters, performs local training, and returns updated parameters (weights+biases of each layer) 
        set_parameters_from_numpy(self.model, parameters)
        self.model.to(self.device)
        
        # Evaluate BEFORE local training (this is the global model performance)
        if not self.initial_eval_done:
            print("\n" + "="*80)
            print(f"[{self.client_name}] INITIAL EVALUATION - BEFORE FEDERATED LEARNING")
            print("="*80)
            self._print_metrics(dataset="train", label="BEFORE FL - Train")
            self._print_metrics(dataset="test", label="BEFORE FL - Test")
            self.initial_eval_done = True
        
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3) # Adam optimizer with learning rate 0.001 
        epochs = int(config.get("epochs", 3))

        self.round_counter += 1
        print(f"\n[{self.client_name}] Starting Local Training - Round {self.round_counter}")

        for _ in range(epochs):
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad() # resets gradients from previous step to prevent accumulation
                y_pred = self.model(X_batch) # forward pass
                loss = self.criterion(y_pred, y_batch) # computes how far off the predictions are from the actual labels using Binary Cross Entropy Loss
                loss.backward() # computes gradients, how much each parameter needs to change to reduce the loss
                optimizer.step() # updates model parameters based on computed gradients

        # Print metrics after local training
        print(f"[{self.client_name}] After Local Training - Round {self.round_counter}")
        self._print_metrics(dataset="train", label=f"Round {self.round_counter} - After Local Training")

        return get_numpy_parameters(self.model), len(self.train_loader.dataset), {} #extracts all weights and biases from your model as numpy arrays, The server then performs Federated Averaging on these parameters.
    
    
    def compute_metrics(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = (y_true == y_pred).mean()
        return accuracy, precision, recall, f1
    
    
    def evaluate(self, parameters, config):
        # Standard Flower evaluation on test data - THIS IS THE GLOBAL MODEL AFTER AGGREGATION
        set_parameters_from_numpy(self.model, parameters)
        self.model.to(self.device)
        self.model.eval()

        X_loader = self.test_loader
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in X_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                preds = (y_pred.cpu().numpy() > 0.5).astype(int).flatten()
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy().flatten())

        accuracy, precision, recall, f1 = self.compute_metrics(
            np.array(all_labels), np.array(all_preds)
        )

        print(f"\n{'='*80}")
        print(f"[{self.client_name}] GLOBAL MODEL EVALUATION - Round {self.round_counter} (AFTER FEDERATED AGGREGATION)")
        print(f"{'='*80}")
        print(f"Test Metrics -> "
              f"Accuracy: {accuracy*100:.2f}%, "
              f"Precision: {precision*100:.2f}%, "
              f"Recall: {recall*100:.2f}%, "
              f"F1: {f1*100:.2f}%\n")

        # Flower expects: loss, num_examples, optional metrics
        loss_sum = self.criterion(torch.tensor(all_preds, dtype=torch.float32),
                                  torch.tensor(all_labels, dtype=torch.float32)).item()
        return float(loss_sum), self.n_test, {"accuracy": float(accuracy)}

    def _print_metrics(self, dataset="train", label=""):
        # Internal helper for printing train/test metrics
        X_loader = self.train_loader if dataset=="train" else self.test_loader
        all_preds, all_labels = [], []

        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in X_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                preds = (y_pred.cpu().numpy() > 0.5).astype(int).flatten()
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy().flatten())

        accuracy, precision, recall, f1 = self.compute_metrics(
            np.array(all_labels), np.array(all_preds)
        )

        print(f"[{self.client_name}] {label} ({dataset.capitalize()}) -> "
              f"Accuracy: {accuracy*100:.2f}%, "
              f"Precision: {precision*100:.2f}%, "
              f"Recall: {recall*100:.2f}%, "
              f"F1: {f1*100:.2f}%")
    



# --------------------- Main ---------------------
if __name__ == "__main__":
    # Hardcoded CSV path and server address
    CSV_PATH = "data\HomeC_labeled.csv"
    SERVER_ADDRESS = "127.0.0.1:8080"
    CLIENT_NAME = "Home C"  # Change this for each client

    # Load dataset
    train_loader, test_loader, n_train, n_test, scaler = load_and_preprocess(CSV_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_features=7)

    client = FlowerClient(model, train_loader, test_loader, device, n_train, n_test, CLIENT_NAME)

    # Start Flower client
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)
