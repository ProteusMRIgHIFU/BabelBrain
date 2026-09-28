'''
Pipeline to execute viscoleastic simulations for TUS experiments

ABOUT:
     author        - Samuel Pichardo
     date          - Nov 22, 2024

'''
from TranscranialModeling.babel_integration.integration_templates import (
    babel_integration_focused_array,
)

class RUN_SIM(babel_integration_focused_array.RUN_SIM):
    pass


class BabelFTD_Simulations(babel_integration_focused_array.BabelFTD_Simulations):
    pass
   

class SimulationConditions(babel_integration_focused_array.SimulationConditions):
    def GenTransducerGeom(self):
        super().GenTransducerGeom(PPWSurface=4)

# Ensures the correct class gets instantiated
BabelFTD_Simulations._SimConditionsClass = SimulationConditions
RUN_SIM._BabelFTDSimClass = BabelFTD_Simulations
