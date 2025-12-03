"""
Simple PQC-FL simulation:
- Initializes FederatedServer
- Creates multiple FederatedClient instances
- Runs a few federated rounds with dummy numpy parameters
"""

import numpy as np

from federated_server import FederatedServer
from federated_client import FederatedClient


def create_dummy_model(num_features: int = 4):
    """Create a tiny dummy 'model' as a dict of numpy arrays."""
    return {
        "W": np.zeros((num_features, 1), dtype=float),
        "b": np.zeros((1,), dtype=float),
    }


def simulate_local_training(global_params, client_id: str):
    """
    Fake local training: return slightly perturbed parameters and num_samples.
    In real code, you would:
      - load local data
      - train using global_params
      - return updated params and sample count
    """
    num_samples = np.random.randint(50, 150)

    updated = {}
    for name, value in global_params.items():
        noise = np.random.normal(loc=0.0, scale=0.01, size=value.shape)
        updated[name] = value + noise

    print(f"[Client {client_id}] Local training done (num_samples={num_samples})")
    return updated, num_samples


def main():
    # ------------------------------------------------------------------
    # 1. Initialize server and global model
    # ------------------------------------------------------------------
    server = FederatedServer(security_level=2)
    initial_model = create_dummy_model(num_features=4)
    server.set_initial_model(initial_model)

    # ------------------------------------------------------------------
    # 2. Create clients and register them
    # ------------------------------------------------------------------
    num_clients = 3
    clients = []
    registry_info = {}

    for cid in range(1, num_clients + 1):
        client_id = f"client{cid}"
        client = FederatedClient(client_id=client_id, security_level=2)
        clients.append(client)

        # Register with server
        registration = client.register_with_server(server_connection=None)
        # registration contains client_id, public_key, timestamp
        resp = server.register_client(
            client_id=registration["client_id"],
            client_public_key=registration["public_key"],
        )
        # resp contains server_kyber_public_key
        registry_info[client_id] = {
            "server_kyber_pk": resp["server_kyber_public_key"],
            "session_key": None,
        }

    # ------------------------------------------------------------------
    # 3. Run a few federated rounds
    # ------------------------------------------------------------------
    num_rounds = 2
    for rnd in range(1, num_rounds + 1):
        print(f"\n========== Federated Round {rnd} ==========")
        global_params = server.get_global_model()

        # Each client trains locally and sends an update
        for client in clients:
            cid = client.client_id
            client_global = global_params  # same model for all clients this round

            updated_params, num_samples = simulate_local_training(client_global, cid)

            # Prepare model_update dict to match FederatedClient.train_local_model() output
            model_update = {
                "client_id": cid,
                "encrypted_gradients": updated_params,  # currently plaintext
                "num_samples": num_samples,
            }

            # First time: use server_kyber_pk; then reuse session_key
            info = registry_info[cid]
            server_kyber_pk = info["server_kyber_pk"]
            session_key = info["session_key"]

            payload = client.secure_send_update(
                model_update=model_update,
                server_kyber_pk=server_kyber_pk,
                session_key=session_key,
            )

            # Store session_key for reuse in next messages
            registry_info[cid]["session_key"] = payload["session_key"]

            # Server receives and processes the update
            server_response = server.receive_update(payload)
            print(f"[Server] Response for {cid}: {server_response}")

        # After receiving all updates, finalize the round
        new_global = server.finalize_round()
        print("\n[Server] New global model parameters:")
        for name, value in new_global.items():
            print(f"  {name}: mean={value.mean():.6f}, std={value.std():.6f}")

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
