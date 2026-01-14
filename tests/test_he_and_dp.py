import numpy as np

from src.homomorphic_encryption import HEManager, HEAggregator
from src.differential_privacy import DifferentialPrivacy


def test_he_public_key_roundtrip():
    he = HEManager(key_size=1024)  # smaller for test speed
    payload = he.serialize_public_key()
    pk2 = HEManager.deserialize_public_key(payload)
    assert int(pk2.n) == int(he.public_key.n)


def test_he_encrypt_serialize_deserialize_add_decrypt():
    he = HEManager(key_size=1024)
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([10.0, 20.0, 30.0])
    enc1 = he.encrypt_vector(vec1)
    enc2 = he.encrypt_vector(vec2)

    p1 = he.serialize_encrypted_vector(enc1)
    p2 = he.serialize_encrypted_vector(enc2)
    dec1 = he.deserialize_encrypted_vector(he.public_key, p1)
    dec2 = he.deserialize_encrypted_vector(he.public_key, p2)

    summed = he.add_encrypted_vectors([dec1, dec2])
    out = he.decrypt_vector(summed, original_shape=(3,))
    assert np.allclose(out, vec1 + vec2, atol=1e-6)


def test_he_aggregator_integer_weighting():
    he = HEManager(key_size=1024)
    agg = HEAggregator(he)
    shapes = {"W": (3,)}

    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([9.0, 10.0, 11.0])
    agg.add_client_update({"W": he.encrypt_vector(v1)}, num_samples=1)
    agg.add_client_update({"W": he.encrypt_vector(v2)}, num_samples=3)
    out = agg.aggregate_and_decrypt(shapes)["W"]

    expected = (v1 * 1 + v2 * 3) / 4.0
    assert np.allclose(out, expected, atol=1e-4)


def test_dp_accounting_is_once_per_step():
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, sensitivity=1.0, noise_type="gaussian")
    x = np.zeros((5,))
    _ = dp.add_noise(x, account=False)
    _ = dp.add_noise(x, account=False)
    assert dp.privacy_spent == 0.0
    dp.account_step()
    assert dp.privacy_spent == 1.0
