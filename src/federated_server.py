from typing import Dict, Any
import numpy as np

from pqc_auth import PQCAuthenticator
from pqc_channel import PQCSecureChannel
from homomorphic_encryption import HEManager, HEAggregator


class FederatedServer:
    """
    Federated learning server with:
    - Kyber-based quantum-safe channels
    - Dilithium-based client authentication
    - Paillier Homomorphic Encryption for privacy-preserving aggregation
    """

    def __init__(self, security_level: int = 2, use_he: bool = True):
        """
        Initialize server-side PQC and HE components.
        
        Args:
            security_level: 2 (medium, default) or 3 (high)
            use_he: Whether to use homomorphic encryption (True = Phase 2)
        """
        # PQC modules
        self.authenticator = PQCAuthenticator(security_level)
        self.secure_channel = PQCSecureChannel(security_level)
        
        # Kyber keypair for server
        kyber_keys = self.secure_channel.server_generate_keypair()
        self.kyber_public_key = kyber_keys["public_key"]
        self.kyber_private_key = kyber_keys["private_key"]
        
        # Homomorphic Encryption
        self.use_he = use_he
        if self.use_he:
            self.he_manager = HEManager(key_size=2048)
            self.he_aggregator = HEAggregator(self.he_manager)
            self.public_he_key_bytes = self.he_manager.serialize_public_key()
            print("✓ Homomorphic Encryption (Paillier) enabled")
        else:
            self.he_manager = None
            self.he_aggregator = None
            self.public_he_key_bytes = None
            print("✓ Homomorphic Encryption disabled (plaintext aggregation)")
        
        # State
        self.clients: Dict[str, Dict[str, Any]] = {}
        self.global_model: Dict[str, np.ndarray] = {}
        self.param_shapes: Dict[str, tuple] = {}
        
        print(f"✓ FederatedServer initialized with PQC (Dilithium + Kyber)")

    def register_client(self, client_id: str, client_public_key: str) -> Dict[str, Any]:
        """
        Register a new client with its Dilithium public key.
        """
        self.clients[client_id] = {
            "dilithium_pk": client_public_key,
            "session_key": None,
            "last_counter": -1,
        }
        
        print(f"✓ Registered client '{client_id}'")
        
        response = {
            "server_kyber_public_key": self.kyber_public_key,
        }
        
        # Include HE public key if enabled
        if self.use_he:
            response["he_public_key"] = self.public_he_key_bytes
        
        return response

    def receive_update(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receive a signed & encrypted update from a client.
        """
        client_id = payload.get("client_id")
        if client_id not in self.clients:
            return {"status": "error", "reason": "unregistered_client"}
        
        client_meta = self.clients[client_id]
        msg_counter = int(payload.get("counter", -1))
        if msg_counter <= client_meta.get("last_counter", -1):
            return {"status": "error", "reason": "replay_detected"}
        
        # 1) Establish/retrieve session key
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
            signed_update = self.secure_channel.decrypt_json(
                encrypted_payload,
                session_key=session_key,
                aad={"client_id": client_id, "counter": msg_counter, "type": "model_update"},
            )
        except Exception as e:
            print(f"✗ Decryption failed for client '{client_id}': {e}")
            return {"status": "error", "reason": "decryption_failed"}
        client_meta["last_counter"] = msg_counter
        
        # 3) Verify Dilithium signature
        public_key_hex = client_meta["dilithium_pk"]
        is_valid = self.authenticator.verify_signature(signed_update, public_key_hex)
        
        if not is_valid:
            print(f"✗ Invalid signature from client '{client_id}'")
            return {"status": "error", "reason": "invalid_signature"}
        
        # 4) Extract model update
        model_update = signed_update["model_update"]
        client_update = model_update.get("model_params")
        num_samples = model_update.get("num_samples", 0)
        
        # 5) Process based on HE mode
        if self.use_he:
            # HE Mode: client_update contains JSON-safe encrypted vectors
            deserialized_update = {}
            for param_name, encrypted_payload_list in client_update.items():
                deserialized_update[param_name] = self.he_manager.deserialize_encrypted_vector(
                    self.he_manager.public_key, encrypted_payload_list
                )
            self.he_aggregator.add_client_update(deserialized_update, num_samples)
        else:
            # Plaintext Mode: client_update contains raw arrays
            self._apply_aggregation(client_update, num_samples)
        
        print(f"✓ Processed update from client '{client_id}' (num_samples={num_samples})")
        
        return {"status": "ok"}

    def _apply_aggregation(self, client_update: Dict[str, Any], num_samples: int) -> None:
        """
        Plaintext aggregation (FedAvg-style) for backwards compatibility.
        """
        if not client_update or num_samples <= 0:
            return
        
        if not hasattr(self, "_agg_sum"):
            self._agg_sum: Dict[str, np.ndarray] = {}
            self._agg_count: float = 0.0
        
        for name, value in client_update.items():
            value = np.array(value)
            if name not in self._agg_sum:
                self._agg_sum[name] = np.zeros_like(value, dtype=float)
            self._agg_sum[name] += value * num_samples
        
        self._agg_count += num_samples

    def finalize_round(self) -> Dict[str, np.ndarray]:
        """
        Compute the new global model by averaging aggregated updates.
        """
        if self.use_he:
            # HE Mode: decrypt aggregated encrypted values
            new_global = self.he_aggregator.aggregate_and_decrypt(self.param_shapes)
            self.global_model = new_global
        else:
            # Plaintext Mode
            if not hasattr(self, "_agg_sum") or self._agg_count == 0:
                return self.global_model
            
            new_global: Dict[str, np.ndarray] = {}
            for name, total in self._agg_sum.items():
                new_global[name] = total / self._agg_count
            
            self.global_model = new_global
            
            del self._agg_sum
            self._agg_count = 0.0
        
        print("✓ Global model updated for this round")
        return self.global_model

    def get_global_model(self) -> Dict[str, np.ndarray]:
        """Return the current global model parameters."""
        return self.global_model

    def set_initial_model(self, initial_params: Dict[str, np.ndarray]) -> None:
        """
        Set the initial global model parameters.
        """
        self.global_model = {k: np.array(v) for k, v in initial_params.items()}
        
        # Store shapes for HE decryption
        self.param_shapes = {k: v.shape for k, v in self.global_model.items()}
        
        print("✓ Initial global model parameters set on server")
