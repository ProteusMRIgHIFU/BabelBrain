
from TranscranialModeling.babel_integration.integration_templates import babel_integration_flat_annular_array


class RUN_SIM(babel_integration_flat_annular_array.RUN_SIM):
    pass


class BabelFTD_Simulations(babel_integration_flat_annular_array.BabelFTD_Simulations):
    pass


class SimulationConditions(babel_integration_flat_annular_array.SimulationConditions):
    pass

# Ensures the correct class gets instantiated
BabelFTD_Simulations._SimConditionsClass = SimulationConditions
RUN_SIM._BabelFTDSimClass = BabelFTD_Simulations
