
from TranscranialModeling.babel_integration.integration_templates import babel_integration_simple_focused


class RUN_SIM(babel_integration_simple_focused.RUN_SIM):
    pass


class BabelFTD_Simulations(babel_integration_simple_focused.BabelFTD_Simulations):
    pass


class SimulationConditions(babel_integration_simple_focused.SimulationConditions):
    pass

# Ensures the correct class gets instantiated
BabelFTD_Simulations._SimConditionsClass = SimulationConditions
RUN_SIM._BabelFTDSimClass = BabelFTD_Simulations
