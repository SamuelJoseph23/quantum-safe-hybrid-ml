import numpy as np

from src.pqc_channel import PQCSecureChannel
from src.pqc_auth import PQCAuthenticator


def test_pqc_channel_encrypt_decrypt_with_aad():
    ch = PQCSecureChannel(security_level=2)
    keys = ch.server_generate_keypair()

    enc = ch.client_encapsulate(keys["public_key"])
    session_key_client = enc["session_key"]
    session_key_server = ch.server_decapsulate(enc["ciphertext"], keys["private_key"])
    assert session_key_client == session_key_server

    msg = b"hello"
    aad = {"client_id": "client_1", "counter": 1, "type": "model_update"}
    encrypted = ch.encrypt_message(msg, session_key_client, aad=ch._aad_bytes(aad))
    decrypted = ch.decrypt_message(encrypted, session_key_server, aad=ch._aad_bytes(aad))
    assert decrypted == msg


def test_pqc_auth_signature_detects_tamper():
    auth = PQCAuthenticator(security_level=2)
    keys = auth.generate_keypair()

    update = {"client_id": "c1", "model_update": {"x": 1}}
    signed = auth.sign_update(update, keys["private_key"])
    assert auth.verify_signature(signed, keys["public_key"]) is True

    # tamper
    signed["model_update"]["model_update"]["x"] = 2
    assert auth.verify_signature(signed, keys["public_key"]) is False
