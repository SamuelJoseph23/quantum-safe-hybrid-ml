import pickle
from typing import Dict, Any
from pqc_auth import PQCAuthenticator
from pqc_channel import PQCSecureChannel

class FederatedClient:
    """
    Federated Learning Client with PQC Security.
    - Authenticates via CRYSTALS-Dilithium.
    - Establishes secure channels via CRYSTALS-Kyber.
    - Encrypts model updates using AES-256 (key derived from Kyber).
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
        
        print(f"✓ Client '{client_id}' initialized (Keys generated)")

    def register_with_server(self, server_address=None) -> Dict[str, str]:
        """
        Prepare registration payload for the server.
        """
        return {
            "client_id": self.client_id,
            "public_key": self.public_key
        }

    def secure_send_update(self, model_update: Dict[str, Any], 
                          server_kyber_pk: str, 
                          session_key: bytes = None) -> Dict[str, Any]:
        """
        Encrypt and sign the model update.
        
        1. If no session key, generating one using Kyber Encapsulation.
        2. Sign the update with Dilithium.
        3. Encrypt the signed package with AES-GCM (Session Key).
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

        # Step 2: Sign the Update (Authentication)
        # We verify integrity of the *plaintext* update before encryption
        signed_package = self.authenticator.sign_update(model_update, self.private_key)
        
        # Step 3: Encrypt the Signed Package (Confidentiality)
        # We serialize the dict to bytes first
        data_bytes = pickle.dumps(signed_package)
        
        encrypted_data = self.secure_channel.encrypt_message(data_bytes, current_session_key)
        
        # Final Payload
        payload_structure['client_id'] = self.client_id
        payload_structure['encrypted_payload'] = encrypted_data
        
        # We return the session key so the main loop can store it for next round
        payload_structure['session_key'] = current_session_key
        
        return payload_structure
