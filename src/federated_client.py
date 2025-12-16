import pickle
from typing import Dict, Any
from pqc_auth import PQCAuthenticator
from pqc_channel import PQCSecureChannel
from homomorphic_encryption import HEManager


class FederatedClient:
    """
    Federated Learning Client with PQC Security and HE support.
    """

    def __init__(self, client_id: str, security_level: int = 2):
        """
        Initialize client PQC components.
        """
        self.client_id = client_id
        
        # Initialize Security Modules
        self.authenticator = PQCAuthenticator(security_level)
        self.secure_channel = PQCSecureChannel(security_level)
        
        # Generate Identity (Dilithium Keypair)
        keys = self.authenticator.generate_keypair()
        self.public_key = keys['public_key']
        self.private_key = keys['private_key']
        
        # HE public key (will be received from server)
        self.he_public_key = None
        
        print(f"✓ Client '{client_id}' initialized (Keys generated)")

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
        print(f"  [Client {self.client_id}] HE public key received")

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
            print(f"  [Client] Performing Kyber Encapsulation...")
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
                encrypted_list = temp_he.encrypt_vector(param_value, self.he_public_key)
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
