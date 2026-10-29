from flask import Blueprint, request, jsonify
from config import Config
from api.consent import DEMO_OTP_INBOX

demo_bp = Blueprint('demo_bp', __name__)

@demo_bp.route('/api/demo/otp-inbox', methods=['GET'])
def get_demo_otp():
    if not Config.DEMO_MODE:
        return jsonify({'error': 'Demo mode is disabled in production.'}), 404

    phone = request.args.get('phone', '').strip()
    if phone and phone in DEMO_OTP_INBOX:
        return jsonify({'messages': DEMO_OTP_INBOX[phone]})
    
    # Return all messages across phones if no query specified
    all_msgs = []
    for ph, msgs in DEMO_OTP_INBOX.items():
        for m in msgs:
            all_msgs.append({**m, 'phone': ph})

    return jsonify({'messages': all_msgs})

