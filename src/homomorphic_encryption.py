"""
Homomorphic Encryption using Paillier (phe library)
Enables privacy-preserving aggregation on encrypted data
Optimized with multiprocessing.
"""

from phe import paillier
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

def _encrypt_chunk(args) -> List[Any]:
    public_key, flat_chunk = args
    return [public_key.encrypt(float(x)) for x in flat_chunk]

def _decrypt_chunk(args) -> List[float]:
    private_key, encrypted_chunk = args
    return [private_key.decrypt(x) for x in encrypted_chunk]

class HEManager:
    """
    Manages Paillier homomorphic encryption for federated learning.
    Paillier supports additive homomorphism (perfect for FedAvg).
    """
    
    def __init__(self, key_size: int = 2048):
        """
        Initialize Paillier encryption.
        """
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_size)
        print(f"[OK] Paillier HE initialized (key_size={key_size} bits)")
        
        # Determine optimal worker count out of CPU pools, max at 4 to avoid overhead
        self.num_workers = min(4, multiprocessing.cpu_count())
    
    def get_public_key(self) -> paillier.PaillierPublicKey:
        return self.public_key
    
    def serialize_public_key(self) -> Dict[str, str]:
        return {"n": str(self.public_key.n)}
    
    @staticmethod
    def deserialize_public_key(key_bytes: Dict[str, str]) -> paillier.PaillierPublicKey:
        if not isinstance(key_bytes, dict):
            raise TypeError("Expected dict-like public key payload")
        n = int(key_bytes["n"])
        return paillier.PaillierPublicKey(n)
    
    def encrypt_vector(self, vector: np.ndarray, public_key: Optional[paillier.PaillierPublicKey] = None) -> List[Any]:
        if public_key is None:
            public_key = self.public_key
        
        if public_key is None:
            raise ValueError("No public key provided for encryption")
        
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        
        flat_vector = vector.flatten()
        
        # Parallel Encryption if vector is large enough
        if len(flat_vector) < 10 or self.num_workers <= 1:
            return [public_key.encrypt(float(x)) for x in flat_vector]
            
        chunk_size = max(1, len(flat_vector) // self.num_workers)
        chunks = [flat_vector[i:i + chunk_size] for i in range(0, len(flat_vector), chunk_size)]
        
        args = [(public_key, c) for c in chunks]
        
        encrypted = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(_encrypt_chunk, args))
            
        for r in results:
            encrypted.extend(r)
            
        return encrypted
    
    def decrypt_vector(self, encrypted_list: List[Any], original_shape: Tuple) -> np.ndarray:
        # Parallel Decryption
        if len(encrypted_list) < 10 or self.num_workers <= 1:
            decrypted = [self.private_key.decrypt(x) for x in encrypted_list]
        else:
            chunk_size = max(1, len(encrypted_list) // self.num_workers)
            chunks = [encrypted_list[i:i + chunk_size] for i in range(0, len(encrypted_list), chunk_size)]
            
            args = [(self.private_key, c) for c in chunks]
            
            decrypted = []
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(_decrypt_chunk, args))
                
            for r in results:
                decrypted.extend(r)
                
        decrypted_array = np.array(decrypted).reshape(original_shape)
        return decrypted_array

    @staticmethod
    def serialize_encrypted_number(enc_num: Any) -> Dict[str, Any]:
        return {"ciphertext": str(enc_num.ciphertext()), "exponent": int(enc_num.exponent)}

    @staticmethod
    def deserialize_encrypted_number(public_key: paillier.PaillierPublicKey, payload: Dict[str, Any]) -> Any:
        ciphertext = int(payload["ciphertext"])
        exponent = int(payload["exponent"])
        return paillier.EncryptedNumber(public_key, ciphertext, exponent)

    def serialize_encrypted_vector(self, encrypted_list: Sequence[Any]) -> List[Dict[str, Any]]:
        return [self.serialize_encrypted_number(x) for x in encrypted_list]

    def deserialize_encrypted_vector(self, public_key: paillier.PaillierPublicKey, payload_list: Sequence[Dict[str, Any]]) -> List[Any]:
        return [self.deserialize_encrypted_number(public_key, p) for p in payload_list]
    
    def add_encrypted_vectors(self, encrypted_lists: List[List[Any]]) -> Optional[List[Any]]:
        if not encrypted_lists:
            return None
        
        result = encrypted_lists[0]
        for enc_vec in encrypted_lists[1:]:
            result = [result[i] + enc_vec[i] for i in range(len(result))]
        return result
    
    def multiply_encrypted_by_scalar(self, encrypted_list: List[Any], scalar: int) -> List[Any]:
        return [x * scalar for x in encrypted_list]


class HEAggregator:
    """
    Helper for federated aggregation with homomorphic encryption.
    """
    
    def __init__(self, he_manager: HEManager):
        self.he_manager = he_manager
        self.encrypted_updates: Dict[str, List[Dict[str, Any]]] = {}
    
    def add_client_update(self, encrypted_params: Dict[str, List[Any]], num_samples: int) -> None:
        for param_name, encrypted_list in encrypted_params.items():
            if param_name not in self.encrypted_updates:
                self.encrypted_updates[param_name] = []
            
            self.encrypted_updates[param_name].append({
                'encrypted': encrypted_list,
                'weight': num_samples
            })
    
    def aggregate_and_decrypt(self, param_shapes: Dict[str, Tuple]) -> Dict[str, np.ndarray]:
        aggregated_params = {}
        
        for param_name, updates in self.encrypted_updates.items():
            total_weight = sum(u['weight'] for u in updates)
            
            weighted_encrypted = []
            for update in updates:
                weight_factor = int(update['weight'])
                weighted = self.he_manager.multiply_encrypted_by_scalar(
                    update['encrypted'], 
                    weight_factor
                )
                weighted_encrypted.append(weighted)
            
            aggregated_encrypted = self.he_manager.add_encrypted_vectors(weighted_encrypted)
            
            if aggregated_encrypted:
                decrypted = self.he_manager.decrypt_vector(
                    aggregated_encrypted, 
                    param_shapes[param_name]
                )
                aggregated_params[param_name] = decrypted / float(total_weight)
        
        self.encrypted_updates = {}
        return aggregated_params
