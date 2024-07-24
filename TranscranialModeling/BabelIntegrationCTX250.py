'''
Pipeline to execute viscoleastic simulations for TUS experiments

ABOUT:
     author        - Samuel Pichardo
     date          - July 24, 2024
     last update   - July 24, 2024

The CTX 250 is identical to the CTX 500, excepting the central frequency
'''
from . import BabelIntegrationCTX500

class RUN_SIM(BabelIntegrationCTX500.RUN_SIM):
    pass
