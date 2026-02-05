"""
Interactive Cybersecurity Demo
A story-driven showcase of the Quantum-Safe Federated Learning System.
"""

import numpy as np
import time
import sys
import os

# Add src directory to path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo_visuals import (
    print_banner, print_status, print_ciphertext_wall,
    print_dp_blur, animate_quantum_attack, animate_mitm_intercept,
    print_comparison, OK, WARN, CRIT, INFO, RESET, BOLD
)
from pqc_auth import PQCAuthenticator
from pqc_channel import PQCSecureChannel
from homomorphic_encryption import HEManager
from differential_privacy import DifferentialPrivacy



def chapter_1_quantum_handshake():
    """Chapter 1: Demonstrate the Kyber key exchange and Dilithium signatures."""
    print_banner("CHAPTER 1: The Quantum-Safe Handshake")
    
    print("  Initializing Post-Quantum Cryptography modules...")
    time.sleep(0.5)
    
    # Initialize PQC components
    channel = PQCSecureChannel(security_level=2)
    auth = PQCAuthenticator(security_level=2)
    
    # Server generates Kyber keypair
    print_status("ok", "Server generating Kyber-768 keypair...")
    server_keys = channel.server_generate_keypair()
    print(f"    Public Key (truncated): {server_keys['public_key'][:32]}...")
    
    # Client encapsulates
    print_status("ok", "Client encapsulating shared secret...")
    encaps = channel.client_encapsulate(server_keys['public_key'])
    print(f"    Ciphertext (truncated): {encaps['ciphertext'][:32]}...")
    
    # Server decapsulates
    print_status("ok", "Server decapsulating to recover session key...")
    session_key = channel.server_decapsulate(encaps['ciphertext'], server_keys['private_key'])
    
    # Verify keys match
    if encaps['session_key'] == session_key:
        print_status("ok", f"Session keys match! ({len(session_key)*8}-bit AES key established)")
    else:
        print_status("critical", "Session key mismatch! Attack detected!")
    
    # Dilithium signature demo
    print()
    print_status("info", "Generating Dilithium identity for client...")
    client_keys = auth.generate_keypair()
    
    test_update = {"client_id": "demo_client", "weights": [0.1, 0.2, 0.3]}
    signed = auth.sign_update(test_update, client_keys['private_key'])
    print_status("ok", f"Update signed! Signature: {signed['signature'][:32]}...")
    
    # --- Interactive MITM Attack ---
    print()
    try:
        choice = input(f"  {WARN}Would you like to attempt a Man-in-the-Middle attack? (y/n): {RESET}").strip().lower()
    except EOFError:
        choice = 'n'
    
    if choice == 'y':
        animate_mitm_intercept()
        print_ciphertext_wall("intercepted", lines=3)
        
        try:
            modify = input(f"  {WARN}Try to modify the intercepted weights? (y/n): {RESET}").strip().lower()
        except EOFError:
            modify = 'n'
        
        if modify == 'y':
            # Tamper with the signed update
            signed['model_update']['weights'] = [9.9, 9.9, 9.9]
            print_status("warn", "Weights modified to [9.9, 9.9, 9.9]!")
            
            # Verify (should fail)
            is_valid = auth.verify_signature(signed, client_keys['public_key'])
            if not is_valid:
                print_status("critical", "DILITHIUM SIGNATURE MISMATCH! Update REJECTED.")
            else:
                print_status("ok", "Signature valid (unexpected).")
        else:
            print_status("info", "No modification attempted.")
    else:
        print_status("info", "Skipping MITM simulation.")
    
    print()
    input("  Press Enter to continue to Chapter 2...")


def chapter_2_he_blindfold():
    """Chapter 2: Demonstrate Homomorphic Encryption (The Server's Blindfold)."""
    print_banner("CHAPTER 2: The Server's Blindfold (Homomorphic Encryption)")
    
    print("  Initializing Paillier Homomorphic Encryption (2048-bit)...")
    he = HEManager(key_size=2048)
    print_status("ok", "HE Keypair generated.")
    
    # Client-side: plaintext weights
    client_weights = np.array([0.25, -0.15, 0.88])
    print(f"\n  {BOLD}Client's Plaintext Weights:{RESET}")
    print(f"    {OK}{client_weights}{RESET}")
    
    # Encrypt
    print_status("info", "Client encrypting weights with HE...")
    encrypted = he.encrypt_vector(client_weights)
    
    # Show server view (ciphertext)
    print(f"\n  {BOLD}Server's View (Encrypted):{RESET}")
    # Get a sample ciphertext representation
    sample_ct = str(encrypted[0].ciphertext())[:80]
    print(f"    {INFO}{sample_ct}...{RESET}")
    print_ciphertext_wall("he_payload", lines=2)
    
    # Demonstrate addition in encrypted space
    print_status("info", "Simulating second client's encrypted weights...")
    client2_weights = np.array([0.10, 0.20, 0.12])
    encrypted2 = he.encrypt_vector(client2_weights)
    
    print_status("info", "Server adding encrypted vectors (BLINDFOLDED)...")
    time.sleep(0.5)
    summed_encrypted = he.add_encrypted_vectors([encrypted, encrypted2])
    print_status("ok", "Encrypted sum computed. Server still sees nothing!")
    
    # Decrypt final result
    print_status("info", "Decrypting final aggregate...")
    decrypted_sum = he.decrypt_vector(summed_encrypted, original_shape=(3,))
    expected_sum = client_weights + client2_weights
    
    print(f"\n  {BOLD}Decrypted Result:{RESET} {OK}{decrypted_sum}{RESET}")
    print(f"  {BOLD}Expected (Plaintext Sum):{RESET} {OK}{expected_sum}{RESET}")
    
    if np.allclose(decrypted_sum, expected_sum, atol=1e-4):
        print_status("ok", "Results match! Privacy-preserving aggregation successful.")
    else:
        print_status("warn", "Minor precision difference (expected with HE).")
    
    print()
    input("  Press Enter to continue to Chapter 3...")


def chapter_3_privacy_blur():
    """Chapter 3: Interactive Differential Privacy Tuner."""
    print_banner("CHAPTER 3: The Privacy Blur (Differential Privacy)")
    
    print("  Differential Privacy adds calibrated noise to protect individual data.")
    print("  Lower epsilon = More noise = More privacy (but less accuracy).")
    print()
    
    while True:
        try:
            user_input = input(f"  {INFO}Enter epsilon value (0.1 - 10.0), or 'q' to quit: {RESET}").strip()
        except EOFError:
            break
        
        if user_input.lower() == 'q':
            break
        
        try:
            epsilon = float(user_input)
            if not (0.1 <= epsilon <= 10.0):
                print_status("warn", "Epsilon out of range. Using 1.0.")
                epsilon = 1.0
        except ValueError:
            print_status("warn", "Invalid input. Using epsilon = 1.0.")
            epsilon = 1.0
        
        # Show visual
        print_dp_blur(epsilon)
        
        # Show simulated accuracy impact
        # Higher epsilon -> closer to baseline accuracy
        baseline_acc = 0.84
        noise_penalty = 0.12 * (1.0 / epsilon)  # Simulated 
        predicted_acc = max(0.50, baseline_acc - noise_penalty)
        print(f"  {INFO}Simulated Accuracy:{RESET} {predicted_acc:.2%} (Baseline: {baseline_acc:.0%})")
        print()
    
    print()
    input("  Press Enter to continue to Chapter 4...")


def chapter_4_quantum_defense():
    """Chapter 4: Quantum Attack Simulation."""
    print_banner("CHAPTER 4: Quantum Attack Defense")
    
    print("  This system uses NIST-standardized Post-Quantum Cryptography.")
    print("  Even if a quantum computer tries to break the encryption, we're ready.")
    print()
    
    try:
        choice = input(f"  {CRIT}Simulate a Quantum Attack? (y/n): {RESET}").strip().lower()
    except EOFError:
        choice = 'n'
    
    if choice == 'y':
        animate_quantum_attack(duration=2.5)
    else:
        print_status("info", "Quantum attack simulation skipped.")
    
    print()
    print_status("ok", "All chapters complete!")
    print_banner("DEMO COMPLETE")
    print("  Your Quantum-Safe Federated Learning System is ready for evaluation.")
    print()


def main():
    """Main entry point for the interactive demo."""
    print_banner("QUANTUM-SAFE FEDERATED LEARNING DEMO")
    print("  Welcome to the Interactive Cybersecurity Demo!")
    print("  This walkthrough demonstrates the security layers protecting your ML system.")
    print()
    input("  Press Enter to begin...")
    
    chapter_1_quantum_handshake()
    chapter_2_he_blindfold()
    chapter_3_privacy_blur()
    chapter_4_quantum_defense()


if __name__ == "__main__":
    main()
