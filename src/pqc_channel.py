"""
Post-Quantum Key Exchange using CRYSTALS-Kyber (ML-KEM)
NIST-standardized quantum-safe KEM
"""

from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
import os
import json
from typing import Any, Dict, Optional

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

    @staticmethod
    def _hkdf_derive(shared_secret: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
        return HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=salt,
            info=info,
        ).derive(shared_secret)

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

        # Derive 256-bit AES key from shared secret via HKDF
        # Salt binds derivation to this handshake ciphertext; info provides protocol context.
        salt = hashes.Hash(hashes.SHA256())
        salt.update(ciphertext)
        salt_bytes = salt.finalize()
        session_key = self._hkdf_derive(shared_secret, salt=salt_bytes, info=b"pqc-fl/aesgcm/v1", length=32)
        
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

        salt = hashes.Hash(hashes.SHA256())
        salt.update(ciphertext)
        salt_bytes = salt.finalize()
        session_key = self._hkdf_derive(shared_secret, salt=salt_bytes, info=b"pqc-fl/aesgcm/v1", length=32)
        
        return session_key

    def encrypt_message(self, message_bytes: bytes, session_key: bytes, aad: Optional[bytes] = None) -> Dict[str, str]:
        """Encrypt bytes with AES-256-GCM using session key (supports AAD)."""
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        aesgcm = AESGCM(session_key)
        ct = aesgcm.encrypt(nonce, message_bytes, aad)
        return {"ciphertext": ct.hex(), "nonce": nonce.hex()}

    def decrypt_message(self, encrypted_data: Dict[str, str], session_key: bytes, aad: Optional[bytes] = None) -> bytes:
        """Decrypt bytes with AES-256-GCM (supports AAD)."""
        ct = bytes.fromhex(encrypted_data["ciphertext"])
        nonce = bytes.fromhex(encrypted_data["nonce"])
        aesgcm = AESGCM(session_key)
        return aesgcm.decrypt(nonce, ct, aad)

    @staticmethod
    def _aad_bytes(aad: Optional[Dict[str, Any]]) -> Optional[bytes]:
        if aad is None:
            return None
        return json.dumps(aad, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def encrypt_json(self, payload: Dict[str, Any], session_key: bytes, aad: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        msg = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return self.encrypt_message(msg, session_key=session_key, aad=self._aad_bytes(aad))

    def decrypt_json(self, encrypted_data: Dict[str, str], session_key: bytes, aad: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        msg = self.decrypt_message(encrypted_data, session_key=session_key, aad=self._aad_bytes(aad))
        return json.loads(msg.decode("utf-8"))
