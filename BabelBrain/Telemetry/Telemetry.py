import requests
import json
import threading
import time
import os
import uuid
import numpy as np


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

def get_install_id(path):

    if os.path.exists(path):
        return open(path).read().strip()

    uid = str(uuid.uuid4())

    with open(path, "w") as f:
        f.write(uid)

    return uid


# --------------------------------------------------
# Main API
# --------------------------------------------------

def send_telemetry(event,
                   idpath=os.path.expanduser("~/.myapp_install_id"),
                   date_sesssion='ids1',
                   APP_NAME = "BabelBrain",
                   APP_VERSION = "1.0.0", 
                   data=[]):


    def _send():
        try:
            #we split data by 5 msgs to avoid ending with a way too wide entry in a single cell
            for n in range(0,len(data),5): 
                subdata=[]
                for k in range(n,np.min((len(data),n+5))):
                    subdata.append(data[k])

                payload = {
                    "schema": 1,
                    "app": APP_NAME,
                    "version": APP_VERSION,
                    "install_id": get_install_id(idpath),
                    "date_sesssion": date_sesssion,
                    "event": event,
                    "data": subdata ,
                }

                post_data = {
                    PAYLOAD_FIELD: json.dumps(payload, ensure_ascii=False)
                }
                requests.post(
                    FORM_URL,
                    data=post_data,
                    timeout=TIMEOUT,
                    allow_redirects=True,
                )
                time.sleep(0.01)

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