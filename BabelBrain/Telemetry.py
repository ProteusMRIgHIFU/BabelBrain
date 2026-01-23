import requests
import json
import threading
import time
import os
import uuid


# Google Forms endpoint
FORM_URL = (
    "https://docs.google.com/forms/d/e/"
    "1FAIpQLSdyaJvAyubH5vRXPDT2neQtN-klL-q26q6sjEi53BrTsvEBFQ"
    "/formResponse"
)

# Google Forms field ID
PAYLOAD_FIELD = "entry.1222718954"

# App metadata
APP_NAME = "MyApp"          # change
APP_VERSION = "1.0.0"       # change

# Network
TIMEOUT = 5.0


# --------------------------------------------------
# Privacy / opt-out
# --------------------------------------------------

def telemetry_enabled():
    return os.getenv("MYAPP_DISABLE_TELEMETRY") != "1"


# --------------------------------------------------
# Stable anonymous install ID
# --------------------------------------------------

def get_install_id():

    path = os.path.expanduser("~/.myapp_install_id")

    if os.path.exists(path):
        return open(path).read().strip()

    uid = str(uuid.uuid4())

    with open(path, "w") as f:
        f.write(uid)

    return uid


# --------------------------------------------------
# Main API
# --------------------------------------------------

def send_telemetry(event, data=None):

    if not telemetry_enabled():
        return

    payload = {
        "schema": 1,
        "app": APP_NAME,
        "version": APP_VERSION,
        "event": event,
        "timestamp": time.time(),
        "install_id": get_install_id(),
        "data": data or {},
    }

    post_data = {
        PAYLOAD_FIELD: json.dumps(payload, ensure_ascii=False)
    }

    def _send():

        try:
            requests.post(
                FORM_URL,
                data=post_data,
                timeout=TIMEOUT,
                allow_redirects=True,
            )

        except Exception:
            # Never affect GUI
            pass


    T=threading.Thread(
        target=_send,
        daemon=True
    )
    T.start()
    T.join()

if __name__ == "__main__":
    print("testing")
    send_telemetry("test_event", {"ok": True})