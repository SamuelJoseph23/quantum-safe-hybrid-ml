"""
Benchmarking Script for Quantum-Safe Federated Learning
Compares performance with and without Homomorphic Encryption
"""

import numpy as np
import json
import os
import time
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import sys

from federated_server import FederatedServer
from federated_client import FederatedClient
from data_utils import load_and_preprocess_data


def train_local_model_sklearn(global_weights, global_intercept, X_local, y_local):
    """Train a local SGDClassifier starting from global parameters."""
    clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, 
                        max_iter=1, tol=None, learning_rate='constant', eta0=0.01,
                        random_state=42)
    
    classes = np.array([0, 1])
    clf.partial_fit(X_local[0:1], y_local[0:1], classes=classes)
    
    clf.coef_ = np.array(global_weights).reshape(1, -1)
    clf.intercept_ = np.array(global_intercept).reshape(1,)
    
    clf.partial_fit(X_local, y_local)
    
    return clf.coef_, clf.intercept_


def evaluate_global_model(server_weights, server_intercept, X_test, y_test):
    """Evaluate the global model on the hold-out test set."""
    clf = SGDClassifier(loss='log_loss', random_state=42)
    clf.partial_fit(X_test[0:1], y_test[0:1], classes=np.array([0, 1]))
    
    clf.coef_ = np.array(server_weights).reshape(1, -1)
    clf.intercept_ = np.array(server_intercept).reshape(1,)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc


def measure_payload_size(payload):
    """
    Measure the size of a payload in bytes.
    """
    import pickle
    return len(pickle.dumps(payload))


def run_experiment(use_he, num_clients=3, num_rounds=5):
    """
    Run a single FL experiment and collect metrics.
    
    Returns:
        dict with performance metrics
    """
    print(f"\n{'='*70}")
    print(f"Running Experiment: HE={'ENABLED' if use_he else 'DISABLED'}")
    print(f"{'='*70}\n")
    
    # Setup Data
    client_data, test_data = load_and_preprocess_data(num_clients)
    X_test, y_test = test_data
    n_features = X_test.shape[1]

    # Initialize Server
    start_server_init = time.time()
    server = FederatedServer(security_level=2, use_he=use_he)
    server_init_time = time.time() - start_server_init
    
    initial_model = {
        "W": np.zeros((1, n_features), dtype=float),
        "b": np.zeros((1,), dtype=float),
    }
    server.set_initial_model(initial_model)

    # Create and Register Clients
    clients = []
    registry_info = {}
    
    start_client_init = time.time()
    for cid in range(num_clients):
        client_id = f"client_{cid+1}"
        client = FederatedClient(client_id=client_id, security_level=2)
        clients.append(client)

        reg_data = client.register_with_server(None)
        resp = server.register_client(reg_data['client_id'], reg_data['public_key'])
        
        registry_info[client_id] = {
            "server_kyber_pk": resp["server_kyber_public_key"],
            "session_key": None
        }
        
        if use_he and "he_public_key" in resp:
            client.set_he_public_key(resp["he_public_key"])
    
    client_init_time = time.time() - start_client_init

    # Training Loop with Metrics
    results = {
        "use_he": use_he,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "server_init_time": server_init_time,
        "client_init_time": client_init_time,
        "rounds": []
    }
    
    total_communication_size = 0
    
    for rnd in range(1, num_rounds + 1):
        print(f"\n=== Round {rnd} ===")
        
        round_start_time = time.time()
        global_params = server.get_global_model()
        global_W = global_params["W"]
        global_b = global_params["b"]

        round_communication_size = 0
        round_encryption_time = 0
        round_training_time = 0
        
        # Client Side
        for i, client in enumerate(clients):
            cid = client.client_id
            X_local, y_local = client_data[i]
            
            # Local Training
            train_start = time.time()
            new_W, new_b = train_local_model_sklearn(global_W, global_b, X_local, y_local)
            train_time = time.time() - train_start
            round_training_time += train_time
            
            # Prepare Update
            model_update = {
                "client_id": cid,
                "encrypted_gradients": {"W": new_W, "b": new_b},
                "num_samples": len(X_local)
            }
            
            # Secure Send (measure time and size)
            info = registry_info[cid]
            
            encrypt_start = time.time()
            payload = client.secure_send_update(
                model_update, 
                info["server_kyber_pk"], 
                info["session_key"],
                use_he=use_he
            )
            encrypt_time = time.time() - encrypt_start
            round_encryption_time += encrypt_time
            
            # Measure payload size
            payload_size = measure_payload_size(payload)
            round_communication_size += payload_size
            
            registry_info[cid]["session_key"] = payload["session_key"]
            
            # Server Receive
            server.receive_update(payload)

        # Server Aggregation
        aggregation_start = time.time()
        new_global_params = server.finalize_round()
        aggregation_time = time.time() - aggregation_start
        
        # Evaluation
        acc = evaluate_global_model(new_global_params["W"], new_global_params["b"], X_test, y_test)
        
        round_total_time = time.time() - round_start_time
        
        print(f"Round {rnd} Accuracy: {acc:.4f} | Time: {round_total_time:.2f}s")
        
        # Store metrics
        results["rounds"].append({
            "round": rnd,
            "accuracy": float(acc),
            "total_time": round_total_time,
            "training_time": round_training_time,
            "encryption_time": round_encryption_time,
            "aggregation_time": aggregation_time,
            "communication_size_bytes": round_communication_size
        })
        
        total_communication_size += round_communication_size
    
    # Calculate summary statistics
    results["summary"] = {
        "final_accuracy": results["rounds"][-1]["accuracy"],
        "avg_round_time": np.mean([r["total_time"] for r in results["rounds"]]),
        "total_training_time": sum([r["training_time"] for r in results["rounds"]]),
        "total_encryption_time": sum([r["encryption_time"] for r in results["rounds"]]),
        "total_aggregation_time": sum([r["aggregation_time"] for r in results["rounds"]]),
        "total_communication_mb": total_communication_size / (1024 * 1024),
        "avg_communication_per_round_kb": (total_communication_size / num_rounds) / 1024
    }
    
    return results


def save_benchmark_results(plaintext_results, he_results, filename='../results/metrics/benchmark_results.json'):
    """Save benchmark comparison results."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "plaintext": plaintext_results,
        "homomorphic_encryption": he_results,
        "comparison": {
            "accuracy_difference": he_results["summary"]["final_accuracy"] - plaintext_results["summary"]["final_accuracy"],
            "time_overhead_percentage": ((he_results["summary"]["avg_round_time"] - plaintext_results["summary"]["avg_round_time"]) / plaintext_results["summary"]["avg_round_time"]) * 100,
            "communication_overhead_percentage": ((he_results["summary"]["total_communication_mb"] - plaintext_results["summary"]["total_communication_mb"]) / plaintext_results["summary"]["total_communication_mb"]) * 100,
            "encryption_overhead_seconds": he_results["summary"]["total_encryption_time"] - plaintext_results["summary"]["total_encryption_time"]
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Benchmark results saved to {filename}")
    return comparison


def create_benchmark_plots(comparison):
    """Create comprehensive benchmark visualization plots."""
    import matplotlib.pyplot as plt
    
    os.makedirs('../results/plots', exist_ok=True)
    
    plaintext = comparison["plaintext"]
    he = comparison["homomorphic_encryption"]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Accuracy Comparison
    ax1 = plt.subplot(2, 3, 1)
    rounds_p = [r["round"] for r in plaintext["rounds"]]
    acc_p = [r["accuracy"] for r in plaintext["rounds"]]
    rounds_he = [r["round"] for r in he["rounds"]]
    acc_he = [r["accuracy"] for r in he["rounds"]]
    
    ax1.plot(rounds_p, acc_p, marker='o', label='Plaintext', linewidth=2, color='#2E86AB')
    ax1.plot(rounds_he, acc_he, marker='s', label='HE (Paillier)', linewidth=2, color='#A23B72')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy: Plaintext vs HE')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Time per Round
    ax2 = plt.subplot(2, 3, 2)
    time_p = [r["total_time"] for r in plaintext["rounds"]]
    time_he = [r["total_time"] for r in he["rounds"]]
    
    x = np.arange(len(rounds_p))
    width = 0.35
    ax2.bar(x - width/2, time_p, width, label='Plaintext', color='#2E86AB')
    ax2.bar(x + width/2, time_he, width, label='HE (Paillier)', color='#A23B72')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Computation Time per Round')
    ax2.set_xticks(x)
    ax2.set_xticklabels(rounds_p)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Communication Size
    ax3 = plt.subplot(2, 3, 3)
    comm_p = [r["communication_size_bytes"]/1024 for r in plaintext["rounds"]]  # KB
    comm_he = [r["communication_size_bytes"]/1024 for r in he["rounds"]]  # KB
    
    ax3.bar(x - width/2, comm_p, width, label='Plaintext', color='#2E86AB')
    ax3.bar(x + width/2, comm_he, width, label='HE (Paillier)', color='#A23B72')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Communication (KB)')
    ax3.set_title('Communication Size per Round')
    ax3.set_xticks(x)
    ax3.set_xticklabels(rounds_p)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Time Breakdown (Stacked Bar)
    ax4 = plt.subplot(2, 3, 4)
    categories = ['Plaintext', 'HE']
    training_times = [plaintext["summary"]["total_training_time"], he["summary"]["total_training_time"]]
    encryption_times = [plaintext["summary"]["total_encryption_time"], he["summary"]["total_encryption_time"]]
    aggregation_times = [plaintext["summary"]["total_aggregation_time"], he["summary"]["total_aggregation_time"]]
    
    ax4.bar(categories, training_times, label='Training', color='#2E86AB')
    ax4.bar(categories, encryption_times, bottom=training_times, label='Encryption', color='#F18F01')
    ax4.bar(categories, aggregation_times, bottom=np.array(training_times)+np.array(encryption_times), 
            label='Aggregation', color='#A23B72')
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Total Time Breakdown')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Overhead Comparison
    ax5 = plt.subplot(2, 3, 5)
    metrics = ['Time\nOverhead', 'Communication\nOverhead']
    overhead_values = [
        comparison["comparison"]["time_overhead_percentage"],
        comparison["comparison"]["communication_overhead_percentage"]
    ]
    colors = ['#F18F01', '#C73E1D']
    bars = ax5.bar(metrics, overhead_values, color=colors)
    ax5.set_ylabel('Overhead (%)')
    ax5.set_title('HE Overhead vs Plaintext')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    # 6. Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    table_data = [
        ['Metric', 'Plaintext', 'HE', 'Overhead'],
        ['Final Accuracy', f"{plaintext['summary']['final_accuracy']:.4f}", 
         f"{he['summary']['final_accuracy']:.4f}", 
         f"{comparison['comparison']['accuracy_difference']:.4f}"],
        ['Avg Round Time', f"{plaintext['summary']['avg_round_time']:.2f}s", 
         f"{he['summary']['avg_round_time']:.2f}s", 
         f"{comparison['comparison']['time_overhead_percentage']:.1f}%"],
        ['Total Comm.', f"{plaintext['summary']['total_communication_mb']:.2f} MB", 
         f"{he['summary']['total_communication_mb']:.2f} MB", 
         f"{comparison['comparison']['communication_overhead_percentage']:.1f}%"],
        ['Encryption Time', f"{plaintext['summary']['total_encryption_time']:.2f}s", 
         f"{he['summary']['total_encryption_time']:.2f}s", 
         f"{comparison['comparison']['encryption_overhead_seconds']:.2f}s"]
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Quantum-Safe Federated Learning: Performance Benchmarks', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('../results/plots/benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Benchmark plot saved to ../results/plots/benchmark_comparison.png")
    plt.close()


def print_summary(comparison):
    """Print a formatted summary of the benchmark results."""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    print("\n📊 ACCURACY:")
    print(f"   Plaintext:  {comparison['plaintext']['summary']['final_accuracy']:.4f}")
    print(f"   HE:         {comparison['homomorphic_encryption']['summary']['final_accuracy']:.4f}")
    print(f"   Difference: {comparison['comparison']['accuracy_difference']:.4f}")
    
    print("\n⏱️  TIME PERFORMANCE:")
    print(f"   Plaintext Avg Round:  {comparison['plaintext']['summary']['avg_round_time']:.2f}s")
    print(f"   HE Avg Round:         {comparison['homomorphic_encryption']['summary']['avg_round_time']:.2f}s")
    print(f"   Overhead:             {comparison['comparison']['time_overhead_percentage']:.1f}%")
    
    print("\n📡 COMMUNICATION:")
    print(f"   Plaintext Total:  {comparison['plaintext']['summary']['total_communication_mb']:.2f} MB")
    print(f"   HE Total:         {comparison['homomorphic_encryption']['summary']['total_communication_mb']:.2f} MB")
    print(f"   Overhead:         {comparison['comparison']['communication_overhead_percentage']:.1f}%")
    
    print("\n🔐 ENCRYPTION OVERHEAD:")
    print(f"   Additional Time: {comparison['comparison']['encryption_overhead_seconds']:.2f}s")
    
    print("\n" + "="*70)


def main():
    """Run complete benchmarking suite."""
    print("\n" + "="*70)
    print("QUANTUM-SAFE FEDERATED LEARNING BENCHMARK")
    print("="*70)
    
    NUM_CLIENTS = 3
    NUM_ROUNDS = 5
    
    # Run Plaintext Experiment
    plaintext_results = run_experiment(use_he=False, num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS)
    
    # Run HE Experiment
    he_results = run_experiment(use_he=True, num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS)
    
    # Save and analyze results
    comparison = save_benchmark_results(plaintext_results, he_results)
    
    # Create visualizations
    create_benchmark_plots(comparison)
    
    # Print summary
    print_summary(comparison)
    
    print("\n✅ Benchmarking complete!")
    print("\n📁 Output files:")
    print("   - results/metrics/benchmark_results.json")
    print("   - results/plots/benchmark_comparison.png")


if __name__ == "__main__":
    main()
