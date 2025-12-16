"""
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
            1: ML_KEM_512, # ~128-bit quantum security
            2: ML_KEM_768, # ~192-bit quantum security
            3: ML_KEM_1024 # ~256-bit quantum security
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
        # NOTE: kyber-py returns (shared_secret, ciphertext) order!
        shared_secret, ciphertext = self.kem.encaps(pk)
        
        # Derive 256-bit AES key from shared secret
        session_key = shared_secret[:32] # Use first 32 bytes for AES-256
        
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
        
        # kyber-py uses decaps(private_key, ciphertext) order!
        shared_secret = self.kem.decaps(sk, ciphertext)
        
        # Derive same 256-bit AES key
        session_key = shared_secret[:32]
        
        return session_key

    def encrypt_message(self, message_bytes, session_key):
        """Encrypt message with AES-256-GCM using session key"""
        iv = os.urandom(12) # 96-bit IV for GCM
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
