from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import os
import base64

class KeyManagement:
    def __init__(self, password):
        self.password = password
        self.salt = os.urandom(16)
        self.key = self.derive_key(password, self.salt)

    def derive_key(self, password, salt):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(password.encode())
        return key

    def encrypt(self, plaintext):
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CFB(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext.encode()) + encryptor.finalize()
        return base64.b64encode(iv + ciphertext).decode()

    def decrypt(self, ciphertext):
        ciphertext = base64.b64decode(ciphertext.encode())
        iv = ciphertext[:16]
        ciphertext = ciphertext[16:]
        cipher = Cipher(algorithms.AES(self.key), modes.CFB(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        return plaintext.decode()

class Authentication:
    def __init__(self, key_management):
        self.key_management = key_management
        self.authenticated_nodes = set()

    def authenticate_node(self, node_id, password):
        if password == "correct_password":  # Replace with a real password check
            self.authenticated_nodes.add(node_id)
            print(f"Node {node_id} authenticated successfully.")
        else:
            print(f"Authentication failed for node {node_id}.")

    def is_node_authenticated(self, node_id):
        return node_id in self.authenticated_nodes

# Example usage
password = "secure_password"
key_management = KeyManagement(password)
authentication = Authentication(key_management)

# Simulate node authentication
authentication.authenticate_node(1, "correct_password")
authentication.authenticate_node(2, "wrong_password")

# Encrypt and decrypt a message
node_id = 1
if authentication.is_node_authenticated(node_id):
    plaintext = "Hello, World!"
    ciphertext = key_management.encrypt(plaintext)
    decrypted_message = key_management.decrypt(ciphertext)
    print(f"Original Message: {plaintext}")
    print(f"Ciphertext: {ciphertext}")
    print(f"Decrypted Message: {decrypted_message}")
else:
    print(f"Node {node_id} is not authenticated.")
