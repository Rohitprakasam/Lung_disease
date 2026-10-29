import os
import shutil
from cryptography.fernet import Fernet
from config import Config

class Storage:
    def __init__(self, storage_dir: str = None, storage_key: str = None):
        self.storage_dir = storage_dir or Config.STORAGE_DIR
        raw_key = storage_key or Config.STORAGE_KEY
        
        # Ensure Fernet key is 32 url-safe base64 encoded bytes
        if isinstance(raw_key, str):
            raw_key = raw_key.encode("utf-8")
        self.cipher = Fernet(raw_key)
        os.makedirs(self.storage_dir, exist_ok=True)

    def get_scan_dir(self, scan_key: str) -> str:
        safe_key = "".join(c for c in scan_key if c.isalnum() or c in ("-", "_"))
        return os.path.join(self.storage_dir, safe_key)

    def save_bytes(self, scan_key: str, filename: str, data: bytes) -> str:
        """
        Encrypts and stores bytes on disk under private storage directory.
        """
        scan_dir = self.get_scan_dir(scan_key)
        os.makedirs(scan_dir, exist_ok=True)
        
        encrypted_data = self.cipher.encrypt(data)
        file_path = os.path.join(scan_dir, f"{filename}.enc")
        with open(file_path, "wb") as f:
            f.write(encrypted_data)
        return file_path

    def load_bytes(self, scan_key: str, filename: str) -> bytes:
        """
        Reads encrypted file from disk and returns decrypted plaintext bytes.
        """
        scan_dir = self.get_scan_dir(scan_key)
        file_path = os.path.join(scan_dir, f"{filename}.enc")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Stored file {filename} not found for scan {scan_key}")
            
        with open(file_path, "rb") as f:
            encrypted_data = f.read()
            
        return self.cipher.decrypt(encrypted_data)

    def delete_scan(self, scan_key: str) -> bool:
        """
        Permanently purges encrypted files for a given scan (used for DPDP right to erasure).
        """
        scan_dir = self.get_scan_dir(scan_key)
        if os.path.exists(scan_dir):
            shutil.rmtree(scan_dir)
            return True
        return False

