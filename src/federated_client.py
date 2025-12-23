import pickle
import numpy as np
from typing import Dict, Any, Union
from pqc_auth import PQCAuthenticator
from pqc_channel import PQCSecureChannel
from homomorphic_encryption import HEManager
from differential_privacy import DifferentialPrivacy

class FederatedClient:
    """
    Federated Learning Client with PQC Security, HE support, and Differential Privacy.
    """
    def __init__(self, client_id: str, security_level: int = 2, 
                 use_dp: bool = False, dp_epsilon: float = 1.0):
        """
        Initialize client PQC components and Privacy mechanism.
        """
        self.client_id = client_id
        
        # Initialize Security Modules
        self.authenticator = PQCAuthenticator(security_level)
        self.secure_channel = PQCSecureChannel(security_level)
        
        # Initialize Differential Privacy (Phase 3A)
        self.use_dp = use_dp
        if self.use_dp:
            # Epsilon = 1.0 is a standard starting point for strong privacy
            self.dp_engine = DifferentialPrivacy(epsilon=dp_epsilon, delta=1e-5)
        
        # Generate Identity (Dilithium Keypair)
        keys = self.authenticator.generate_keypair()
        self.public_key = keys['public_key']
        self.private_key = keys['private_key']
        
        # HE public key (will be received from server)
        self.he_public_key = None
        
        print(f"✓ Client '{client_id}' initialized (DP: {self.use_dp})")

    def register_with_server(self, server_address=None) -> Dict[str, str]:
        """
        Prepare registration payload for the server.
        """
        return {
            "client_id": self.client_id,
            "public_key": self.public_key
        }

    def set_he_public_key(self, he_public_key_bytes):
        """
        Receive and store HE public key from server.
        """
        self.he_public_key = HEManager.deserialize_public_key(he_public_key_bytes)
        # print(f" [Client {self.client_id}] HE public key received")

    def apply_differential_privacy(self, 
                                 local_weights: np.ndarray, 
                                 global_weights: np.ndarray, 
                                 clipping_norm: float = 1.5) -> np.ndarray:
        """
        Phase 3A: Apply Local Differential Privacy (LDP) to weight updates.
        
        Mechanism:
        1. Calculate Delta: ΔW = W_local - W_global
        2. Clip Delta: Ensure L2 norm <= clipping_norm
        3. Add Noise: Add Laplace noise scaled by sensitivity/epsilon
        4. Reconstruct: W_new = W_global + ΔW_clipped + Noise
        """
        if not self.use_dp:
            return local_weights

        # 1. Compute Update (Delta)
        delta = local_weights - global_weights
        
        # 2. Gradient Clipping (L2 Norm)
        # We manually clip here before passing to the DP engine which adds noise
        delta_norm = np.linalg.norm(delta)
        if delta_norm > clipping_norm:
            scaling_factor = clipping_norm / delta_norm
            delta = delta * scaling_factor
            
        # 3. Add Noise using the DP Engine
        # The engine handles the noise generation based on epsilon
        noisy_delta = self.dp_engine.add_noise(delta)
        
        # 4. Return differentially private weights
        return global_weights + noisy_delta

    def secure_send_update(self, model_update: Dict[str, Any], 
                          server_kyber_pk: str, 
                          session_key: bytes = None,
                          use_he: bool = True) -> Dict[str, Any]:
        """
        Encrypt and sign the model update.
        
        Args:
            use_he: If True, encrypt gradients with Paillier before sending
        """
        
        payload_structure = {}
        current_session_key = session_key
        
        # Step 1: Establish Secure Channel (if needed)
        if current_session_key is None:
            # print(f" [Client] Performing Kyber Encapsulation...")
            encaps_result = self.secure_channel.client_encapsulate(server_kyber_pk)
            current_session_key = encaps_result['session_key']
            payload_structure['kyber_ciphertext'] = encaps_result['ciphertext']
        else:
            payload_structure['kyber_ciphertext'] = None

        # Step 1.5: Apply Homomorphic Encryption if enabled
        if use_he and self.he_public_key:
            # Create a temporary HE manager for encryption (no private key needed)
            temp_he = HEManager.__new__(HEManager)
            temp_he.public_key = self.he_public_key
            
            # Encrypt gradients with Paillier
            encrypted_gradients = {}
            for param_name, param_value in model_update['encrypted_gradients'].items():
                
                # --- FIX: Ensure input is always a NumPy array ---
                if isinstance(param_value, np.ndarray):
                    vec = param_value
                else:
                    vec = np.array(param_value)
                # -----------------------------------------------
                    
                encrypted_list = temp_he.encrypt_vector(vec, self.he_public_key)
                # Serialize the encrypted list for transmission
                encrypted_gradients[param_name] = pickle.dumps(encrypted_list)
            
            # Replace plaintext gradients with encrypted ones
            model_update_to_send = {
                "client_id": model_update["client_id"],
                "encrypted_gradients": encrypted_gradients,
                "num_samples": model_update["num_samples"]
            }
        else:
            model_update_to_send = model_update

        # Step 2: Sign the Update (Authentication)
        signed_package = self.authenticator.sign_update(model_update_to_send, self.private_key)
        
        # Step 3: Encrypt the Signed Package (Confidentiality)
        data_bytes = pickle.dumps(signed_package)
        encrypted_data = self.secure_channel.encrypt_message(data_bytes, current_session_key)
        
        # Final Payload
        payload_structure['client_id'] = self.client_id
        payload_structure['encrypted_payload'] = encrypted_data
        payload_structure['session_key'] = current_session_key
        
        return payload_structure
