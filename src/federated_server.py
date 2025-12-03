import pickle
from typing import Dict, Any

import numpy as np

from pqc_auth import PQCAuthenticator
from pqc_channel import PQCSecureChannel


class FederatedServer:
    """
    Minimal federated learning server with:
    - Kyber-based quantum-safe channels
    - Dilithium-based client authentication
    - Plaintext model aggregation (for now)
    """

    def __init__(self, security_level: int = 2):
        """
        Initialize server-side PQC components and state.

        Args:
            security_level: 2 (medium, default) or 3 (high)
        """
        # PQC modules
        self.authenticator = PQCAuthenticator(security_level)
        self.secure_channel = PQCSecureChannel(security_level)

        # Kyber keypair for server
        kyber_keys = self.secure_channel.server_generate_keypair()
        self.kyber_public_key = kyber_keys["public_key"]
        self.kyber_private_key = kyber_keys["private_key"]

        # State
        self.clients: Dict[str, Dict[str, Any]] = {}  # client_id -> {dilithium_pk, session_key}
        self.global_model: Dict[str, np.ndarray] = {}  # param_name -> np.ndarray

        print("✓ FederatedServer initialized with PQC (Dilithium + Kyber)")

    # ------------------------------------------------------------------
    # Client registration and key management
    # ------------------------------------------------------------------
    def register_client(self, client_id: str, client_public_key: str) -> Dict[str, Any]:
        """
        Register a new client with its Dilithium public key.

        Args:
            client_id: Unique identifier for the client
            client_public_key: Client's Dilithium public key (hex string)

        Returns:
            dict containing server Kyber public key for channel setup
        """
        self.clients[client_id] = {
            "dilithium_pk": client_public_key,
            "session_key": None,  # filled after first update
        }
        print(f"✓ Registered client '{client_id}'")

        return {
            "server_kyber_public_key": self.kyber_public_key,
        }

    # ------------------------------------------------------------------
    # Receiving and processing updates
    # ------------------------------------------------------------------
    def receive_update(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receive a signed & encrypted update from a client.

        Expected payload fields (from FederatedClient.secure_send_update):
          - client_id
          - kyber_ciphertext (None if reusing existing session)
          - encrypted_payload (AES-GCM ciphertext, iv, tag)
          - session_key (client-side copy; ignored here)

        Returns:
            dict with status info
        """
        client_id = payload.get("client_id")
        if client_id not in self.clients:
            return {"status": "error", "reason": "unregistered_client"}

        client_meta = self.clients[client_id]

        # 1) If this is the first message, derive session_key via Kyber decapsulation
        kyber_ciphertext = payload.get("kyber_ciphertext")
        if kyber_ciphertext is not None:
            session_key = self.secure_channel.server_decapsulate(
                kyber_ciphertext, self.kyber_private_key
            )
            client_meta["session_key"] = session_key
            print(f"✓ Session key established for client '{client_id}'")
        else:
            session_key = client_meta.get("session_key")
            if session_key is None:
                return {"status": "error", "reason": "no_session_key"}

        # 2) Decrypt the signed update with AES-GCM
        encrypted_payload = payload.get("encrypted_payload")
        try:
            signed_bytes = self.secure_channel.decrypt_message(
                encrypted_payload, session_key
            )
        except Exception as e:
            print(f"✗ Decryption failed for client '{client_id}': {e}")
            return {"status": "error", "reason": "decryption_failed"}

        signed_update = pickle.loads(signed_bytes)

        # 3) Verify Dilithium signature
        public_key_hex = client_meta["dilithium_pk"]
        is_valid = self.authenticator.verify_signature(signed_update, public_key_hex)
        if not is_valid:
            print(f"✗ Invalid signature from client '{client_id}'")
            return {"status": "error", "reason": "invalid_signature"}

        # 4) Extract model update
        model_update = signed_update["model_update"]
        client_update = model_update.get("encrypted_gradients")  # currently plaintext
        num_samples = model_update.get("num_samples", 0)

        # For now, treat encrypted_gradients as plaintext gradients/weights dict.
        self._apply_aggregation(client_update, num_samples)

        print(
            f"✓ Processed update from client '{client_id}' "
            f"(num_samples={num_samples})"
        )
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Aggregation logic (plaintext FedAvg-style)
    # ------------------------------------------------------------------
    def _apply_aggregation(self, client_update: Dict[str, Any], num_samples: int) -> None:
        """
        Aggregate client model updates into the global model (FedAvg-style).

        Args:
            client_update: Dict of model parameter arrays (currently plaintext)
            num_samples: Number of local samples used for this update
        """
        if not client_update or num_samples <= 0:
            return

        # Initialize aggregation buffers if needed
        if not hasattr(self, "_agg_sum"):
            self._agg_sum: Dict[str, np.ndarray] = {}
            self._agg_count: float = 0.0

        # Weighted sum
        for name, value in client_update.items():
            value = np.array(value)
            if name not in self._agg_sum:
                self._agg_sum[name] = np.zeros_like(value, dtype=float)
            self._agg_sum[name] += value * num_samples

        self._agg_count += num_samples

    def finalize_round(self) -> Dict[str, np.ndarray]:
        """
        Compute the new global model by averaging aggregated updates.

        Returns:
            global_model: Dict of parameter_name -> numpy array
        """
        if not hasattr(self, "_agg_sum") or self._agg_count == 0:
            # No updates this round; return existing model
            return self.global_model

        new_global: Dict[str, np.ndarray] = {}
        for name, total in self._agg_sum.items():
            new_global[name] = total / self._agg_count

        self.global_model = new_global

        # Clear aggregation buffers
        del self._agg_sum
        self._agg_count = 0.0

        print("✓ Global model updated for this round")
        return self.global_model

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_global_model(self) -> Dict[str, np.ndarray]:
        """Return the current global model parameters."""
        return self.global_model

    def set_initial_model(self, initial_params: Dict[str, np.ndarray]) -> None:
        """
        Set the initial global model parameters (e.g., from baseline.py).

        Args:
            initial_params: Dict of parameter_name -> numpy array
        """
        self.global_model = {k: np.array(v) for k, v in initial_params.items()}
        print("✓ Initial global model parameters set on server")
