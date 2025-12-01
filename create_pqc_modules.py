"""
Script to create PQC Python module files for quantum-safe-hybrid-ml/src/
Run this from the quantum-safe-hybrid-ml directory
"""
import os

def create_pqc_modules():
    """Create the three PQC Python modules in src/ folder"""

    # Ensure we're creating in the right location
    src_folder = "src"

    if not os.path.exists(src_folder):
        print(f"Creating {src_folder}/ directory...")
        os.makedirs(src_folder, exist_ok=True)

    print("=" * 70)
    print("CREATING PQC MODULE FILES")
    print("=" * 70)

    # ===== FILE 1: pqc_auth.py =====
    pqc_auth_code = '''"""
Post-Quantum Authentication using CRYSTALS-Dilithium (ML-DSA)
NIST FIPS 204 compliant digital signatures
"""
from dilithium_py.ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
import hashlib
import json

class PQCAuthenticator:
    """Handles participant authentication with Dilithium signatures"""

    def __init__(self, security_level=2):
        """
        Initialize Dilithium authenticator

        Args:
            security_level: 2 (ML-DSA-44), 3 (ML-DSA-65), 5 (ML-DSA-87)
                           Higher = more secure but slower
        """
        self.dilithium_schemes = {
            2: ML_DSA_44,  # ~128-bit quantum security
            3: ML_DSA_65,  # ~192-bit quantum security
            5: ML_DSA_87   # ~256-bit quantum security
        }
        self.scheme = self.dilithium_schemes.get(security_level, ML_DSA_44)

    def generate_keypair(self):
        """Generate Dilithium public/private key pair"""
        pk, sk = self.scheme.keygen()
        return {
            'public_key': pk.hex(),
            'private_key': sk.hex()
        }

    def sign_update(self, model_update, private_key_hex):
        """
        Sign encrypted model update with Dilithium

        Args:
            model_update: dict containing encrypted gradients
            private_key_hex: participant's private key (hex string)

        Returns:
            dict with signature attached
        """
        # Serialize model update
        update_bytes = json.dumps(model_update, sort_keys=True).encode()

        # Convert hex key back to bytes
        sk = bytes.fromhex(private_key_hex)

        # Generate signature
        signature = self.scheme.sign(sk, update_bytes)

        return {
            'model_update': model_update,
            'signature': signature.hex(),
            'hash': hashlib.sha256(update_bytes).hexdigest()
        }

    def verify_signature(self, signed_update, public_key_hex):
        """
        Verify Dilithium signature on model update

        Args:
            signed_update: dict with model_update and signature
            public_key_hex: participant's public key (hex string)

        Returns:
            bool: True if signature valid, False otherwise
        """
        try:
            # Extract components
            model_update = signed_update['model_update']
            signature = bytes.fromhex(signed_update['signature'])
            pk = bytes.fromhex(public_key_hex)

            # Recreate message
            update_bytes = json.dumps(model_update, sort_keys=True).encode()

            # Verify signature
            is_valid = self.scheme.verify(pk, update_bytes, signature)

            # Also verify hash integrity
            expected_hash = hashlib.sha256(update_bytes).hexdigest()
            hash_valid = (expected_hash == signed_update['hash'])

            return is_valid and hash_valid

        except Exception as e:
            print(f"Signature verification failed: {e}")
            return False
'''

    # ===== FILE 2: pqc_channel.py =====
    pqc_channel_code = '''"""
Post-Quantum Key Exchange using CRYSTALS-Kyber (ML-KEM)
NIST-standardized quantum-safe KEM
"""
from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

class PQCSecureChannel:
    """Establishes quantum-safe communication channels with Kyber"""

    def __init__(self, security_level=2):
        """
        Initialize Kyber KEM

        Args:
            security_level: 1 (ML-KEM-512), 2 (ML-KEM-768), 3 (ML-KEM-1024)
        """
        self.kem_schemes = {
            1: ML_KEM_512,   # ~128-bit quantum security
            2: ML_KEM_768,   # ~192-bit quantum security
            3: ML_KEM_1024   # ~256-bit quantum security
        }
        self.kem = self.kem_schemes.get(security_level, ML_KEM_768)

    def server_generate_keypair(self):
        """Server generates Kyber KEM keypair"""
        pk, sk = self.kem.keygen()
        return {
            'public_key': pk.hex(),
            'private_key': sk.hex()
        }

    def client_encapsulate(self, server_public_key_hex):
        """
        Client encapsulates shared secret using server's public key

        Args:
            server_public_key_hex: Server's Kyber public key

        Returns:
            dict with ciphertext and derived session key
        """
        pk = bytes.fromhex(server_public_key_hex)

        # Encapsulate: generates shared secret + ciphertext
        ciphertext, shared_secret = self.kem.encaps(pk)

        # Derive 256-bit AES key from shared secret
        session_key = shared_secret[:32]  # Use first 32 bytes for AES-256

        return {
            'ciphertext': ciphertext.hex(),
            'session_key': session_key
        }

    def server_decapsulate(self, ciphertext_hex, server_private_key_hex):
        """
        Server decapsulates to recover shared secret

        Args:
            ciphertext_hex: Ciphertext from client
            server_private_key_hex: Server's private key

        Returns:
            bytes: Derived session key (same as client's)
        """
        ciphertext = bytes.fromhex(ciphertext_hex)
        sk = bytes.fromhex(server_private_key_hex)

        # Decapsulate to recover shared secret
        shared_secret = self.kem.decaps(ciphertext, sk)

        # Derive same 256-bit AES key
        session_key = shared_secret[:32]

        return session_key

    def encrypt_message(self, message_bytes, session_key):
        """Encrypt message with AES-256-GCM using session key"""
        iv = os.urandom(12)  # 96-bit IV for GCM
        cipher = Cipher(
            algorithms.AES(session_key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(message_bytes) + encryptor.finalize()

        return {
            'ciphertext': ciphertext.hex(),
            'iv': iv.hex(),
            'tag': encryptor.tag.hex()
        }

    def decrypt_message(self, encrypted_data, session_key):
        """Decrypt message with AES-256-GCM"""
        ciphertext = bytes.fromhex(encrypted_data['ciphertext'])
        iv = bytes.fromhex(encrypted_data['iv'])
        tag = bytes.fromhex(encrypted_data['tag'])

        cipher = Cipher(
            algorithms.AES(session_key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        return plaintext
'''

    # ===== FILE 3: federated_client.py =====
    federated_client_code = '''"""
Federated Learning Client with PQC Authentication & Encryption
"""
import numpy as np
from pqc_auth import PQCAuthenticator
from pqc_channel import PQCSecureChannel
# from encryption import HybridEncryption  # Uncomment when you have this module
import pickle

class FederatedClient:
    """FL client with post-quantum security"""

    def __init__(self, client_id, security_level=2):
        self.client_id = client_id

        # Initialize PQC modules
        self.authenticator = PQCAuthenticator(security_level)
        self.secure_channel = PQCSecureChannel(security_level)

        # Generate authentication keypair
        self.auth_keys = self.authenticator.generate_keypair()

        # Your existing hybrid encryption (uncomment when ready)
        # self.hybrid_enc = HybridEncryption()

        print(f"✓ Client {client_id} initialized with PQC security")

    def register_with_server(self, server_connection):
        """
        Register with FL server and establish secure channel

        Returns:
            dict: Registration info including public key
        """
        registration = {
            'client_id': self.client_id,
            'public_key': self.auth_keys['public_key'],
            'timestamp': str(np.datetime64('now'))
        }

        return registration

    def train_local_model(self, local_data, global_model_params):
        """
        Train on local encrypted data

        Args:
            local_data: Client's private dataset
            global_model_params: Current global model from server

        Returns:
            dict: Encrypted model updates
        """
        # Classify sensitive vs non-sensitive features
        sensitive_features = self._identify_sensitive_features(local_data)

        # Encrypt data with hybrid approach
        # encrypted_data = self.hybrid_enc.encrypt_dataset(
        #     local_data, 
        #     sensitive_features
        # )

        # Train model on encrypted data (simplified placeholder)
        # In practice, use your encrypted_train.py logic
        gradients = self._compute_encrypted_gradients(
            local_data,  # Use encrypted_data when HybridEncryption is available
            global_model_params
        )

        return {
            'client_id': self.client_id,
            'encrypted_gradients': gradients,
            'num_samples': len(local_data) if hasattr(local_data, '__len__') else 0
        }

    def secure_send_update(self, model_update, server_kyber_pk, session_key=None):
        """
        Sign update with Dilithium & encrypt channel with Kyber

        Args:
            model_update: Encrypted gradients
            server_kyber_pk: Server's Kyber public key
            session_key: Existing session key (optional)

        Returns:
            dict: Signed and encrypted update ready to send
        """
        # Step 1: Sign the update with Dilithium
        signed_update = self.authenticator.sign_update(
            model_update,
            self.auth_keys['private_key']
        )

        # Step 2: Establish/use secure channel with Kyber
        if session_key is None:
            # First communication: establish channel
            kyber_result = self.secure_channel.client_encapsulate(server_kyber_pk)
            session_key = kyber_result['session_key']
            kyber_ciphertext = kyber_result['ciphertext']
        else:
            # Reuse existing session
            kyber_ciphertext = None

        # Step 3: Encrypt the signed update
        signed_bytes = pickle.dumps(signed_update)
        encrypted_payload = self.secure_channel.encrypt_message(
            signed_bytes,
            session_key
        )

        return {
            'client_id': self.client_id,
            'kyber_ciphertext': kyber_ciphertext,  # None if reusing session
            'encrypted_payload': encrypted_payload,
            'session_key': session_key  # Store for reuse (keep secret!)
        }

    def _identify_sensitive_features(self, data):
        """Identify which features need HE vs AES"""
        # Placeholder - implement based on your data schema
        # Return list of sensitive feature indices
        return []

    def _compute_encrypted_gradients(self, encrypted_data, model_params):
        """Compute gradients on encrypted data"""
        # Placeholder - implement your encrypted training logic
        # Return encrypted gradients as dict
        return {'gradients': 'placeholder'}
'''

    # Write all three files
    files = {
        'pqc_auth.py': pqc_auth_code,
        'pqc_channel.py': pqc_channel_code,
        'federated_client.py': federated_client_code
    }

    for filename, code in files.items():
        filepath = os.path.join(src_folder, filename)
        with open(filepath, 'w') as f:
            f.write(code)
        print(f"✓ Created: src/{filename}")

    print("\n" + "=" * 70)
    print("PQC MODULES CREATED SUCCESSFULLY!")
    print("=" * 70)
    print("\nFiles created in src/:")
    print("  • pqc_auth.py       - Dilithium authentication")
    print("  • pqc_channel.py    - Kyber secure channels")
    print("  • federated_client.py - FL client with PQC")
    print("\nNext steps:")
    print("  1. Install dependencies: pip install dilithium-py kyber-py cryptography")
    print("  2. Test the modules with a simple demo")
    print("  3. Integrate with your existing encryption.py module")

if __name__ == "__main__":
    create_pqc_modules()
