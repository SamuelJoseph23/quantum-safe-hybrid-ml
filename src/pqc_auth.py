"""
Post-Quantum Authentication using CRYSTALS-Dilithium (ML-DSA)
NIST FIPS 204 compliant digital signatures
"""

from dilithium_py.ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
import hashlib
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """
    Helper to convert NumPy arrays to lists for JSON serialization.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class PQCAuthenticator:
    """Handles participant authentication with Dilithium signatures"""

    def __init__(self, security_level=2):
        """
        Initialize Dilithium authenticator
        Args:
            security_level: 2 (ML-DSA-44), 3 (ML-DSA-65), 5 (ML-DSA-87)
        """
        self.dilithium_schemes = {
            2: ML_DSA_44, # ~128-bit quantum security
            3: ML_DSA_65, # ~192-bit quantum security
            5: ML_DSA_87  # ~256-bit quantum security
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
        # Serialize model update using our custom NumpyEncoder
        update_bytes = json.dumps(model_update, sort_keys=True, cls=NumpyEncoder).encode()
        
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
            
            # Recreate message using the SAME encoder
            update_bytes = json.dumps(model_update, sort_keys=True, cls=NumpyEncoder).encode()
            
            # Verify signature
            is_valid = self.scheme.verify(pk, update_bytes, signature)
            
            # Also verify hash integrity
            expected_hash = hashlib.sha256(update_bytes).hexdigest()
            hash_valid = (expected_hash == signed_update['hash'])
            
            return is_valid and hash_valid
            
        except Exception as e:
            print(f"Signature verification failed: {e}")
            return False
