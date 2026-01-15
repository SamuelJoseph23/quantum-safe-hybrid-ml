"""
Homomorphic Encryption using Paillier (phe library)
Enables privacy-preserving aggregation on encrypted data
"""

from phe import paillier
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple


class HEManager:
    """
    Manages Paillier homomorphic encryption for federated learning.
    Paillier supports additive homomorphism (perfect for FedAvg).
    """
    
    def __init__(self, key_size=2048):
        """
        Initialize Paillier encryption.
        
        Args:
            key_size: Key size in bits (2048 = standard, 3072 = high security)
        """
        # Generate Paillier keypair
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_size)
        print(f"[OK] Paillier HE initialized (key_size={key_size} bits)")
    
    def get_public_key(self):
        """
        Return public key for clients to encrypt.
        """
        return self.public_key
    
    def serialize_public_key(self):
        """
        Serialize public key for transmission to clients.
        """
        # NOTE: Do NOT use pickle for anything that could be "network-facing".
        # Paillier public key is fully described by modulus n.
        return {"n": str(self.public_key.n)}
    
    @staticmethod
    def deserialize_public_key(key_bytes):
        """
        Deserialize public key received from server.
        """
        if isinstance(key_bytes, (bytes, bytearray)):
            raise TypeError("Expected dict-like public key payload, got bytes.")
        n = int(key_bytes["n"])
        return paillier.PaillierPublicKey(n)
    
    def encrypt_vector(self, vector, public_key=None):
        """
        Encrypt a numpy array using Paillier.
        
        Args:
            vector: numpy array to encrypt
            public_key: Public key (if None, uses self.public_key)
        
        Returns:
            List of encrypted values
        """
        if public_key is None:
            public_key = self.public_key
        
        if public_key is None:
            raise ValueError("No public key provided for encryption")
        
        # Ensure input is numpy array
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        
        # Flatten if multi-dimensional
        flat_vector = vector.flatten()
        
        # Encrypt each value
        encrypted = [public_key.encrypt(float(x)) for x in flat_vector]
        
        return encrypted
    
    def decrypt_vector(self, encrypted_list, original_shape):
        """
        Decrypt a list of Paillier ciphertexts back to numpy array.
        
        Args:
            encrypted_list: List of encrypted values
            original_shape: Shape to reshape decrypted values into
        
        Returns:
            Decrypted numpy array
        """
        # Decrypt each value
        decrypted = [self.private_key.decrypt(x) for x in encrypted_list]
        
        # Convert to numpy and reshape
        decrypted_array = np.array(decrypted).reshape(original_shape)
        
        return decrypted_array

    @staticmethod
    def serialize_encrypted_number(enc_num: Any) -> Dict[str, Any]:
        """
        Serialize a phe Paillier EncryptedNumber into a JSON-safe dict.
        """
        # phe EncryptedNumber exposes `ciphertext()` and `exponent`
        return {"ciphertext": str(enc_num.ciphertext()), "exponent": int(enc_num.exponent)}

    @staticmethod
    def deserialize_encrypted_number(public_key: Any, payload: Dict[str, Any]) -> Any:
        """
        Reconstruct a phe Paillier EncryptedNumber from JSON-safe dict.
        """
        ciphertext = int(payload["ciphertext"])
        exponent = int(payload["exponent"])
        # phe supports constructing EncryptedNumber via paillier.EncryptedNumber(public_key, ciphertext, exponent)
        return paillier.EncryptedNumber(public_key, ciphertext, exponent)

    def serialize_encrypted_vector(self, encrypted_list: Sequence[Any]) -> List[Dict[str, Any]]:
        return [self.serialize_encrypted_number(x) for x in encrypted_list]

    def deserialize_encrypted_vector(self, public_key: Any, payload_list: Sequence[Dict[str, Any]]) -> List[Any]:
        return [self.deserialize_encrypted_number(public_key, p) for p in payload_list]
    
    def add_encrypted_vectors(self, encrypted_lists):
        """
        Homomorphically add multiple encrypted vectors.
        
        Args:
            encrypted_lists: List of encrypted vectors (each is a list of EncryptedNumbers)
        
        Returns:
            Aggregated encrypted vector
        """
        if not encrypted_lists:
            return None
        
        # Start with first vector
        result = encrypted_lists[0]
        
        # Add remaining vectors homomorphically
        for enc_vec in encrypted_lists[1:]:
            result = [result[i] + enc_vec[i] for i in range(len(result))]
        
        return result
    
    def multiply_encrypted_by_scalar(self, encrypted_list, scalar):
        """
        Multiply encrypted vector by a plaintext scalar.
        
        Args:
            encrypted_list: List of encrypted values
            scalar: Plaintext scalar
        
        Returns:
            Result encrypted vector
        """
        return [x * scalar for x in encrypted_list]


class HEAggregator:
    """
    Helper for federated aggregation with homomorphic encryption.
    """
    
    def __init__(self, he_manager):
        """
        Args:
            he_manager: HEManager instance with the private key (server-side)
        """
        self.he_manager = he_manager
        self.encrypted_updates = {}  # param_name -> list of (encrypted_list, weight)
    
    def add_client_update(self, encrypted_params, num_samples):
        """
        Add a client's encrypted update to the aggregation buffer.
        
        Args:
            encrypted_params: Dict of {param_name: encrypted_list}
            num_samples: Number of samples used for weighting
        """
        for param_name, encrypted_list in encrypted_params.items():
            if param_name not in self.encrypted_updates:
                self.encrypted_updates[param_name] = []
            
            # Store encrypted update with its weight
            self.encrypted_updates[param_name].append({
                'encrypted': encrypted_list,
                'weight': num_samples
            })
    
    def aggregate_and_decrypt(self, param_shapes):
        """
        Perform weighted aggregation on encrypted values and decrypt result.
        
        Args:
            param_shapes: Dict of {param_name: original_shape}
        
        Returns:
            Dict of {param_name: decrypted_numpy_array}
        """
        aggregated_params = {}
        
        for param_name, updates in self.encrypted_updates.items():
            # Calculate total weight
            total_weight = sum(u['weight'] for u in updates)
            
            # Weighted aggregation (integer weights):
            # - Multiply ciphertexts by num_samples (integer) homomorphically
            # - Sum all ciphertext vectors
            # - Decrypt and divide by total_weight in plaintext
            weighted_encrypted = []
            for update in updates:
                weight_factor = int(update['weight'])
                weighted = self.he_manager.multiply_encrypted_by_scalar(
                    update['encrypted'], 
                    weight_factor
                )
                weighted_encrypted.append(weighted)
            
            # Homomorphically add all weighted updates
            aggregated_encrypted = self.he_manager.add_encrypted_vectors(weighted_encrypted)
            
            # Decrypt final result
            decrypted = self.he_manager.decrypt_vector(
                aggregated_encrypted, 
                param_shapes[param_name]
            )
            aggregated_params[param_name] = decrypted / float(total_weight)
        
        # Clear buffer
        self.encrypted_updates = {}
        
        return aggregated_params
