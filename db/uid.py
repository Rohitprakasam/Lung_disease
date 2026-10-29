import secrets

# Crockford Base32 alphabet (excludes I, L, O, U to prevent visual confusion)
CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
CHAR_TO_VAL = {c: i for i, c in enumerate(CROCKFORD_ALPHABET)}

# Canonical normalization map (handles lowercase, visual confusions i/l->1, o->0)
NORM_MAP = {
    'i': '1', 'I': '1',
    'l': '1', 'L': '1',
    'o': '0', 'O': '0'
}

def _compute_check_char(chars: str) -> str:
    """Computes a weighted modulo-32 check character."""
    total = 0
    for idx, char in enumerate(chars):
        val = CHAR_TO_VAL[char]
        total += val * (idx + 1)
    return CROCKFORD_ALPHABET[total % 32]

def generate_patient_uid() -> str:
    """
    Generates a cryptographically strong Patient UID with check character.
    Format: PT-XXXX-XXXX (e.g. PT-7K3M-9Q2D)
    """
    raw_chars = "".join(secrets.choice(CROCKFORD_ALPHABET) for _ in range(7))
    check_char = _compute_check_char(raw_chars)
    all_chars = raw_chars + check_char # 8 chars total
    return f"PT-{all_chars[:4]}-{all_chars[4:]}"

def validate_patient_uid(uid: str) -> bool:
    """
    Validates a patient UID against format and check character integrity.
    """
    if not isinstance(uid, str):
        return False
    
    cleaned = uid.strip().upper().replace(" ", "").replace("-", "")
    if cleaned.startswith("PT"):
        cleaned = cleaned[2:]
        
    if len(cleaned) != 8:
        return False

    normalized = []
    for c in cleaned:
        c_norm = NORM_MAP.get(c, c)
        if c_norm not in CHAR_TO_VAL:
            return False
        normalized.append(c_norm)

    body = "".join(normalized[:7])
    expected_check = _compute_check_char(body)
    actual_check = normalized[7]

    return expected_check == actual_check

