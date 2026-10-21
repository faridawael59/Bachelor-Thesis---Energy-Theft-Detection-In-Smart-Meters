import flwr as fl

import torch
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl

import flwr as fl
import matplotlib.pyplot as plt


# Custom strategy to track losses
class FedAvgWithHistory(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.losses = []
        
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation losses and track them."""
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated is not None:
            loss_aggregated, metrics_aggregated = aggregated
            self.losses.append(loss_aggregated)
            print(f"Round {server_round} - Aggregated Loss: {loss_aggregated:.4f}")
        
        return aggregated


if __name__ == "__main__":
    strategy = FedAvgWithHistory(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
     
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=12),
        strategy=strategy,
    )
    
        # After training completes, plot the loss curve
if strategy.losses:
    rounds = list(range(1, len(strategy.losses) + 1))
        
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, strategy.losses, marker='o', linewidth=2, markersize=8, color='blue')
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('BCE Loss', fontsize=12)
    plt.title('Federated Learning - Loss Over Rounds', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rounds)
    plt.tight_layout()
        
        # Save the plot
    plt.savefig('federated_loss_curve.png', dpi=300, bbox_inches='tight')
    print(f"\nLoss curve saved as 'federated_loss_curve.png'")
        
        # Print summary
    print(f"\nTraining Summary:")
    print(f"Initial Loss (Round 1): {strategy.losses[0]:.4f}")
    print(f"Final Loss (Round 12): {strategy.losses[-1]:.4f}")
    print(f"Total Loss Reduction: {strategy.losses[0] - strategy.losses[-1]:.4f}")
        
    plt.show()


