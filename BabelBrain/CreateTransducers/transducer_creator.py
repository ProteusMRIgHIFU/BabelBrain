import importlib
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from BabelViscoFDTD.tools.RayleighAndBHTE import (
    ForwardSimple,
    InitCuda,
    InitMetal,
    InitOpenCL,
    SpeedofSoundWater,
)
from jinja2 import Environment, FileSystemLoader
from PySide6.QtWidgets import QDialog, QMessageBox, QVBoxLayout

from CreateTransducers.transducer_verification_dialog import (
    PlotWidget,
    TransducerVerificationDialog,
)
from RunServerCalculation import RAYLEIGH_TEST, RunServerCalculation
from Utils.paths import resource_path
from Utils.transducer_registry import DEFAULT_TRANSDUCER_NAMES

logger = logging.getLogger(__name__)

class YAMLParameterError(Exception):
    def __init__(self, message, full_key=None):
        self.full_key = full_key
        super().__init__(message)

# =============================================================================
# CONSTANTS
# =============================================================================

COORD_VARS = {'cartesian': ('x', 'y', 'z'), 'spherical': ('r', 'theta', 'phi')}
CUSTOM_TRANSDUCERS_FOLDER = Path.home() / '.config' / 'BabelBrain' / 'Transducers'
TX_GEOMETRIES = {
    "simple_focused": {
        "annular": False,
        "coordinate_system": "cartesian",
        "flat": True,
        "spherical": False,
        "steering_axes": None,
    },
    "flat_annular_array": {
        "annular": True,
        "coordinate_system": "spherical",
        "flat": True,
        "spherical": False,
        "steering_axes": {"z"},  # can only steer along depth axis
    },
    "focused_annular_array": {
        "annular": True,
        "coordinate_system": "spherical",
        "flat": False,
        "spherical": True,
        "steering_axes": {"z"},  # can only steer along depth axis
    },
    "flat_array_2D": {
        "annular": False,
        "coordinate_system": "cartesian",
        "flat": True,
        "spherical": False,
        "steering_axes": {"x", "y", "z"},  # full 3D steering
    },
    "focused_array": {
        "annular": False,
        "coordinate_system": ("cartesian", "spherical"),  # user-selectable
        "flat": False,
        "spherical": True,
        "steering_axes": {"x", "y", "z"},  # full 3D steering
    },
}
VALID_FREQUENCIES = range(200000,1005000,5000)

# =============================================================================
# Helper Functions
# =============================================================================

def get_class_name(name):
    # Convert dashes between numbers to underscores.
    name = re.sub(r"(?<=\d)-(?=\d)", "_", name)

    # Split on underscores/dashes unless they are between two numbers.
    parts = re.split(r"(?<!\d)[_-]+|[_-]+(?!\d)", name)

    pascalcase_name = "".join(part[:1].upper() + part[1:] for part in parts if part)

    return pascalcase_name

# =============================================================================
# Dialogs / Message Boxes / Widgets
# =============================================================================

def overwrite_msgbox(tx_name) -> None:
    message_box = QMessageBox()
    message_box.setIcon(QMessageBox.Icon.Warning)
    message_box.setWindowTitle("Overwrite Existing Files")
    message_box.setText(f"Transducer files already exist for {tx_name}.\n\nDo you want to overwrite them?")
    message_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    message_box.setDefaultButton(QMessageBox.StandardButton.Yes)
    return message_box

def restore_msgbox() -> None:
    message_box = QMessageBox()
    message_box.setWindowTitle("New Transducer Creation Cancelled")
    message_box.setText("Restored previous version of transducer")
    return message_box

# =============================================================================
# Main Class
# =============================================================================


class CustomTransducer:

    def __init__(self, bb_version: str, transducer_yaml: str = '', gpu: str = '', computing_backend: str = '', remote_server: dict | None = None) -> None:
        
        # Initial Values
        self.aperture_size: float | None = None
        self.bb_version = bb_version
        self.computing_backend = computing_backend
        self.coordinate_system: str | None = None
        self.coordinate_vars: list[str] = []
        self.distance_outplane: float | None = None
        self.elements: dict | None = None
        self.element_size: float | None = None
        self.frequencies: list[int] = []
        self.focal_length: float | None = None
        self.geometry_type: str | None = None
        self.gpu = gpu
        self.is_gpu_initialized = False
        self.is_annular: bool = False
        self.is_spherical: bool = False
        self.is_steerable: bool = False
        self.name: str | None = None
        self.num_elements: int | None = None
        self.old_tx_temp_dir: str = ''
        self.PlanTUS: dict | None = None
        self.remote_server = remote_server
        self.yaml_file = transducer_yaml
        self.yaml_line_info: dict | None = None
        self.rings: dict | None = None
        self.steering_axes: set = set()
        self.xsteering_limits: list | None = None
        self.ysteering_limits: list | None = None
        self.zsteering_limits: list | None = None
        
        try:
            # Load/Validate transducer details
            try:
                tx_params = self.load_custom_tx_config_file(self.yaml_file)
                self.yaml_line_info = self._get_yaml_line_numbers()
                self._validate_custom_tx_params(tx_params)
            except Exception as error:
                raise self._format_yaml_input_error(error) from error

            print("Custom transducer file loading and validation complete")
            
            # Create transducer files needed for operation
            self._create_tx_files()
            
            # Validate newly created transducer
            self._validate_tx()
            
        except Exception as error:
            
            # Remove newly created tx files if exception occurs
            if hasattr(self, "tx_folder") and not self.old_tx_temp_dir:
                if self.tx_folder.exists() and self.tx_folder.is_dir():
                    print("Deleting newly created transducer files")
                    
                    shutil.rmtree(self.tx_folder)
                    print(f"Deleted: {self.tx_folder} and its contents")
                
            raise error
    
    # =========================================================================
    # YAML ERROR REPORTING
    # =========================================================================

    def _get_yaml_line_numbers(self):

        with open(self.yaml_file, "r") as f:
            lines = f.readlines()

        root = yaml.compose("".join(lines))

        line_numbers = {}

        def walk(node, path=""):

            if isinstance(node, yaml.MappingNode):

                for key_node, value_node in node.value:

                    key = key_node.value
                    current_path = f"{path}.{key}" if path else key

                    # PyYAML line numbers are zero-based
                    line_number = key_node.start_mark.line + 1

                    line_numbers[current_path] = {
                        "line": line_number,
                        "text": lines[line_number - 1].rstrip("\n"),
                    }

                    walk(value_node, current_path)

            elif isinstance(node, yaml.SequenceNode):

                for i, item_node in enumerate(node.value):

                    current_path = f"{path}[{i}]"

                    # PyYAML line numbers are zero-based
                    line_number = item_node.start_mark.line + 1

                    line_numbers[current_path] = {
                        "line": line_number,
                        "text": lines[line_number - 1].rstrip("\n"),
                    }

                    walk(item_node, current_path)

        walk(root)

        return line_numbers

    def _format_yaml_input_error(self, error) -> ValueError:
        """
        Add the YAML source row to an input/validation error before it reaches
        the GUI error dialog.
        """
        if not isinstance(error,YAMLParameterError) or "missing" in str(error):
            return error
        
        error_message = str(error).strip()

        line = self.yaml_line_info[error.full_key]["line"]
        line_text = self.yaml_line_info[error.full_key]["text"]

        if isinstance(error, yaml.MarkedYAMLError):
            problem = error.problem or error_message
            error_message = f"YAML error on line {line}:\nyaml_line\n\n{problem}"
        else:
            error_message = f"YAML validation error on line {line}:yaml_line\n\nReason:\n{error_message}"

        # Include the source row itself so the user can immediately see the
        # YAML entry that should be inspected or corrected.
        if line_text:
            # error_message += f"\n\n{line_text.strip()}"
            error_message = error_message.replace("yaml_line","\n"+line_text.strip())
        else:
            error_message = error_message.replace("yaml_line","\n"+line_text.strip())

        return ValueError(error_message)

    # =========================================================================
    # FILE LOADING
    # =========================================================================
    
    def load_custom_tx_config_file(self, tx_yaml: str) -> dict:

        print("Loading custom transducer file")
        
        # Read yaml then save.return tx_params dict
        if not os.path.isfile(tx_yaml):
            raise ValueError(f'{tx_yaml} does not exist')
        
        with open(tx_yaml, 'r') as file:
            custom_tx_params = yaml.safe_load(file)
            
        # Record custom tx template version
        self.template_version = self._get_template_version(tx_yaml)

        return custom_tx_params
    
    def _get_template_version(self, tx_yaml: str) -> str:
        version = ''
        with open(tx_yaml, "r", encoding="utf-8") as f:
            for line in f:
                version_found = re.search(r"(?<=Template Version: )(\d|\.)*",line)
                
                if version_found:
                    version = version_found[0]
                    break

        if not version:
            raise ValueError("Template version number is missing from your custom transducer yaml file")
        
        return version
    
    # =========================================================================
    # PARAMETER VALIDATION PIPELINE
    # =========================================================================
    
    def _validate_custom_tx_params(self, tx_params: dict) -> None:
        print("Validating custom transducer file")
        
        # Validation order matters: geometry and num_elements must be set before
        # downstream validators (e.g. _validate_elements) that depend on them.
        self._validate_name(tx_params)                                                                          # sets: self.name
        self._validate_geometry(tx_params)                                                                      # sets: self.geometry_type, self.is_annular, ...
        self._validate_frequencies(tx_params)                                                                   # sets: self.frequencies
        self._validate_positive_param('aperture_size', (int, float), tx_params, unit="m")                       # sets: self.aperture_size
        self._validate_positive_param('focal_length',  (int, float), tx_params, unit="m")                       # sets: self.focal_length
        self._validate_positive_param('distance_outplane', (int, float), tx_params, allow_zero=True, unit="m")  # sets: self.distance_outplane
        if self.geometry_type != "simple_focused":
            self._validate_positive_param('num_elements',  int, tx_params)                                      # sets: self.num_elements
            if self.geometry_type in ['flat_array_2D','focused_array']:
                self._validate_positive_param('element_size', (int, float), tx_params)                          # sets: self.element_size
        else:
            self.num_elements = 1
        self._validate_coordinate_system(tx_params)                                                             # sets: self.coordinate_system, self.coordinate_vars
        self._validate_elements(tx_params)                                                                      # sets: self.elements
        self._validate_annular(tx_params)                                                                       # sets: self.rings
        self._validate_steering(tx_params)                                                                      # sets: self.xsteering_limits, self.ysteering_limits, self.zsteering_limits
        self._validate_PlanTUS(tx_params)                                                                       # sets: self.PlanTUS
    
    # ---------------------------------------------------------------------
    # INTERNAL HELPERS (GENERIC)
    # ---------------------------------------------------------------------
    
    def _get_param(self, key: str, expected_type: type | tuple[type, ...], param_dict: dict, optional: bool = False, parent_key: str = "", list_type: type | tuple[type, ...] | None = None):
        """
        Helper function to ensure key exists in dict and the value is the correct type
        
        Args:
            key (str): key to be checked in param_dict.
            expected_type (type): expected type of param_dict[key].
            param_dict (dict): dict containing values.
            optional (bool): Ignores missing key error if True.
            list_type (type): Expected type of entries if param_dict[key] is a list.
        
        Returns:
            val: value of param_dict[<key>]
        
        Raises:
            ValueError: If key does not exist in param_dict or it's value type does not match expected_type
        """
        parent_key_text = ''
        if parent_key:
            parent_key_text = parent_key + '.'
            
        # Check key exists
        if key not in param_dict.keys():
            if not optional:
                raise YAMLParameterError(
                    f"The following parameter is missing from the custom transducer yaml: {parent_key_text}{key}",
                    full_key=f"{parent_key_text}{key}"
                )
            else:
                return
        
        # Check value type
        val = param_dict[key]
        if not isinstance(val, expected_type):
            type_name = expected_type.__name__ if isinstance(expected_type, type) else " or ".join(t.__name__ for t in expected_type)
            raise YAMLParameterError(
                f"{parent_key_text}{key} was not specified as {type_name} in custom transducer yaml file",
                full_key=f"{parent_key_text}{key}"
            )
        
        # Check types inside list
        if list_type is not None and isinstance(val, list):
            type_name = list_type.__name__ if isinstance(list_type, type) else " or ".join(t.__name__ for t in list_type)
            for i, item in enumerate(val):
                if not isinstance(item, list_type):
                    raise YAMLParameterError(
                        f"{parent_key_text}{key}[{i}] ({item}) was not specified as {type_name} in custom transducer yaml file",
                        full_key=f"{parent_key_text}{key}[{i}]"
                    )
        
        # Return a copy for mutable types to prevent accidental mutation of the original yaml data
        if isinstance(val, (list, dict)):
            return val.copy()
        else:
            return val
    
    def _validate_positive_param(self, key: str, expected_type: type | tuple[type, ...], tx_params: dict, allow_zero: bool = False, unit: str = "") -> None:
        """
        Validates that a transducer parameter exists, is the correct type, and is positive.

        Args:
            key (str): Parameter name to look up in tx_params.
            expected_type (type or tuple of types): Expected type(s) for the parameter value.
            tx_params (dict): Raw transducer parameters loaded from yaml file.
            allow_zero (bool): If True, accepts values >= 0. If False, requires values > 0. Defaults to False.
            unit (str): Unit of parameter

        Raises:
            ValueError: If the parameter is missing, not the expected type, or fails the
                        positivity check.

        Sets:
            self.<key> (expected_type): Validated parameter value, converted to float if
                                        expected_type is not int.
        """
        val = self._get_param(key, expected_type, tx_params)
        
        # Check value is positive
        if allow_zero and val < 0:
            raise YAMLParameterError(
                f"{key} ({val} {unit}) must be >= 0 {unit}",
                full_key=f"{key}"
            )
        elif not allow_zero and val <= 0:
            raise YAMLParameterError(
                f"{key} ({val} {unit}) must be > 0",
                full_key=f"{key}"
            )
        
        # Ensure value is float if that is expected type
        result = float(val) if expected_type is not int else val
        
        # Assign to self
        setattr(self,key,result) # Equivalent to self.<key> = result
        print(f"Transducer {key}: {result} {unit}")
    
    def _validate_numeric_list_dict( self, param_dict: dict, num_elements: int | None = None, context_name: str = "parameter", allow_negative: bool = True) -> None:
        """
        Validates that all lists match the expected length and contain valid values.

        Args:
            param_dict (dict): Dictionary mapping parameter names to lists of values.
            num_elements (int): Expected length of each list, typically self.num_elements.
            context_name (str): Human-readable label for param_dict used in error messages
                                (e.g. 'elements', 'annular').
            allow_negative (bool): Set to False if element values should be positive

        Raises:
            ValueError: If any list length does not match num_elements or contains
            invalid negative values.
        """
    
        # Collect all invalid entries before raising so the user sees every problem at once
        bad_entries = []
        for key, values in param_dict.items():
            if num_elements is not None and len(values) != num_elements:
                raise ValueError(f"Number of entries in {key} ({len(values)}) does not match num_elements ({num_elements})")
            
            if not allow_negative:
                for i, val in enumerate(values):
                    if val < 0:
                        bad_entries.append(f"   {key}[{i}]: {val!r} (negative values not allowed)")
        
        if bad_entries:
            raise ValueError(f"{context_name} contains invalid entries:\n" + "\n".join(bad_entries))
    
    def _validate_limits(self, limits: list, context_name: str = "") -> None:
        """
        Validate limits supplied in list
        
        Args:
            limits (list): [min_value max_value]
            context_name (str): Optional string to provide more detail to error message
        
        Raises:
            ValueError: If max_value is less than min_value
        """
        if len(limits) != 2:
            raise YAMLParameterError(
                f"{context_name} limits must have exactly 2 entries [min, max], got {len(limits)}",
                full_key=context_name
            )
        

        min_limit = limits[0]
        max_limit = limits[1]
        if min_limit > max_limit:
            raise YAMLParameterError(
                f"{context_name}: Min value ({min_limit}) must be less than max value ({max_limit})",
                full_key=context_name
            )
    
    # ---------------------------------------------------------------------
    # FIELD VALIDATORS
    # ---------------------------------------------------------------------
    
    def _validate_name(self, tx_params: dict) -> None:
        """
        Validates the transducer name parameter.
        
        Args:
            tx_params (dict): Raw transducer parameters loaded from yaml file.
        
        Raises:
            ValueError: If name is missing, not valid type, contains spaces, 
                        special characters, or does not begin with a letter.
        
        Sets:
            self.name (str): Validated transducer name.
        """
        tx_name = self._get_param('name', str, tx_params)

        if not re.match(r'^[a-zA-Z]', tx_name):
            raise YAMLParameterError("Transducer name must begin with a letter", 'name')
        
        if re.search(r'\s', tx_name):
            raise YAMLParameterError("Transducer name cannot contain spaces", 'name')
        
        special_chars = set(re.findall(r'[^a-zA-Z0-9_-]', tx_name))
        if special_chars:
            raise YAMLParameterError(f"Transducer name cannot contain special characters ({', '.join(special_chars)})", 'name')
        
        self.name = tx_name
        self.class_name = get_class_name(self.name)
        
        if self.class_name in DEFAULT_TRANSDUCER_NAMES:
            raise YAMLParameterError(
                'You cannot overwrite default transducers, please enter a different name for your transducer',
                full_key='name'
            )
            
        print(f"Transducer Name: {tx_name}\nTransducer Class Name: {self.class_name}")
    
    def _validate_geometry(self, tx_params: dict) -> None:
        """
        Validates the transducer geometry_type parameter.
        
        Args:
            tx_params (dict): Raw transducer parameters loaded from yaml file.
        
        Raises:
            ValueError: If geometry_type is missing, not valid type, or isn't valid choice
        
        Sets:
            self.geometry_type (str): Validated transducer geometry_type
            self.is_annular (bool): Boolean indicating geometry is of annular variation
            self.is_flat (bool): Boolean indicating geometry is flat
            self.is_spherical (bool): Boolean indicating geometry is spherical
            self.is_steerable (bool): Boolean indicating if geometry type allows electronic steering of focus
            self.steering_axes (tuple): Tuple indicating axes which have steering capabilities
        """
        
        # Geometry type validation
        tx_geometry_type = self._get_param('geometry_type', str, tx_params)
        if tx_geometry_type not in TX_GEOMETRIES.keys():
            valid_geoms_str = "\n".join(TX_GEOMETRIES.keys())
            raise YAMLParameterError(
                f"{tx_geometry_type} is not a valid geometry choice. Expecting one of the following:\n\n{valid_geoms_str}",
                full_key='geometry_type'
            )
        self.geometry_type = tx_geometry_type
        print(f"Transducer Geometry: {tx_geometry_type}")
        
        # Property assignments
        tx_steering_axes = TX_GEOMETRIES[tx_geometry_type]['steering_axes']
        if tx_steering_axes is not None:
            self.is_steerable = True
            self.steering_axes = tx_steering_axes
            
        self.is_annular = TX_GEOMETRIES[tx_geometry_type]['annular']
        self.is_flat = TX_GEOMETRIES[tx_geometry_type]['flat']
        self.is_spherical = TX_GEOMETRIES[tx_geometry_type]['spherical']
    
    def _validate_frequencies(self, tx_params: dict) -> None:
        """
        Validates the transducer frequencies parameter.
        
        Args:
            tx_params (dict): Raw transducer parameters loaded from yaml file.
        
        Raises:
            ValueError: If frequencies is missing, not valid type (int or float), 
            not in valid range (200-1000kHz), or isn't at valid frequency step (5kHz)
        
        Sets:
            self.frequencies (list): Validated transducer frequencies
        """
        
        tx_frequencies = self._get_param('frequencies', list, tx_params, list_type=(int,float))
        print("Transducer Frequencies:")
        for i,freq in enumerate(tx_frequencies):
            
            # Ensure no decimal points in frequency
            if not freq.is_integer():
                raise YAMLParameterError(
                    f"Invalid specified frequency ({freq} Hz), frequency must be an integer value",
                    full_key=f'frequencies[{i}]'
                )
            
            # Ensure frequency is at valid step in frequency range
            int_freq = int(freq)
            if int_freq not in VALID_FREQUENCIES:
                raise YAMLParameterError(
                    f"Invalid specified frequency ({int_freq} Hz), frequency must be at a 5kHz interval value within the 200-1000 kHz range",
                    full_key=f'frequencies[{i}]'
                )

            # Add valid frequency to list
            self.frequencies.append(int_freq)
            print(f"   {int_freq} Hz")
        
        # Check for duplicate frequencies 
        if len(tx_frequencies) != len(set(tx_frequencies)):
            raise ValueError("frequencies list contains duplicate entries")

    def _validate_coordinate_system(self, tx_params: dict) -> None:
        """
        Validates the transducer element_coordinate_system parameter.
        
        Args:
            tx_params (dict): Raw transducer parameters loaded from yaml file.
        
        Raises:
            ValueError: If element_coordinate_system is missing, not valid type, or isn't valid choice
        
        Sets:
            self.coordinate_system (str): Validated transducer element coordinate system
            self.coordinate_vars (list): List of dimension variable names (x,y,z for cartesian or r,theta,phi for spherical)
        """
        # Geometries other than focused array have a fixed coordinate system — no user input needed
        if self.geometry_type != "focused_array":
            self.coordinate_system = TX_GEOMETRIES[self.geometry_type]['coordinate_system']
            self.coordinate_vars = COORD_VARS[self.coordinate_system]
            print(f"Transducer Coordinate System: {self.coordinate_system}")
            print(f"Transducer Coordinate Variables: {self.coordinate_vars}")
            return
        
        # Spherical geometries allow user to choose coordinate system
        tx_coordinate_system = self._get_param('element_coordinate_system', str, tx_params)
        
        # Validate user specified coordinate system
        valid_tx_coordinate_systems = TX_GEOMETRIES[self.geometry_type]['coordinate_system']
        if tx_coordinate_system not in valid_tx_coordinate_systems:
            valid_coord_systems_str = "\n".join(valid_tx_coordinate_systems)
            raise YAMLParameterError(
                f"{tx_coordinate_system} is not a valid coordinate system choice. Expecting one of the following:\n\n{valid_coord_systems_str}",
                full_key='element_coordinate_system'
            )
        
        # Assign properties
        self.coordinate_system = tx_coordinate_system
        self.coordinate_vars = COORD_VARS[tx_coordinate_system]
        print(f"Transducer Coordinate System: {tx_coordinate_system}")
        print(f"Transducer Coordinate Variables: {self.coordinate_vars}")
         
    def _validate_elements(self, tx_params: dict) -> None:
        """
        Validates the transducer elements parameter.
        
        Args:
            tx_params (dict): Raw transducer parameters loaded from yaml file.
        
        Raises:
            ValueError: If elements or any of its subcomponents are missing or not valid type. If there is a mismatch in
                        number of sub elements and the num_element parameter. If element position do not make sense physically.
        
        Sets:
            self.elements (dict): Validated transducer elements.
        """
        # Element coordinates do not need to be specified for transducers not capabable of xy steering
        if len(self.steering_axes) != 3:
            return
        
        tx_elements = self._get_param('elements', dict, tx_params)
        for dim_var in self.coordinate_vars:
            _ = self._get_param(dim_var, list, tx_elements, parent_key='elements', list_type=(int,float))
        self._validate_numeric_list_dict(tx_elements,self.num_elements,'elements')
        
        self.elements = tx_elements
        for dim_key,dim_values in tx_elements.items():
            logger.debug(f"Transducer Element {dim_key} Values:\n{dim_values}")
    
    def _validate_annular(self, tx_params: dict) -> None:
        """
        Validates the transducer annular parameter.
        
        Args:
            tx_params (dict): Raw transducer parameters loaded from yaml file.
        
        Raises:
            ValueError: If annular or any of its subcomponents are missing or not valid type. If there is a mismatch in
                        number of rings and the num_element parameter. If ring diameters do not make sense physically.
        
        Sets:
            self.rings (dict): Validated transducer ring diameters.
        """
        if not self.is_annular:
            return
        
        tx_rings = self._get_param('annular', dict, tx_params)
        tx_rings_new = {}
        inner_diameters = self._get_param('inner_ring_diameters', list, tx_rings, parent_key='annular', list_type=(int,float))
        outer_diameters = self._get_param('outer_ring_diameters', list, tx_rings, parent_key='annular', list_type=(int,float))
        self._validate_numeric_list_dict(tx_rings,self.num_elements,'annular',allow_negative=False)
        
        # Check outer ring is always bigger than inner ring
        bad_entries = []
        for i, (inner, outer) in enumerate(zip(inner_diameters, outer_diameters)):
            if outer <= inner:
                bad_entries.append(f"inner_ring_diameters[{i}] ({inner}) > outer_ring_diameters[{i}] ({outer})")
        if bad_entries:
            raise YAMLParameterError(
                f"inner_ring_diameters cannot be bigger than corresponding outer_ring_diameter:\n{bad_entries}",
                full_key='annular'
            )
            
        # Rename keys
        tx_rings_new["inner_diameters"] = inner_diameters
        tx_rings_new["outer_diameters"] = outer_diameters
        self.rings = tx_rings_new
        
        # Convert to mm for human-readable logging (yaml values are in metres)
        inner_diams_mm = [d * 1e3 for d in inner_diameters]
        outer_diams_mm = [d * 1e3 for d in outer_diameters]
        print(f"Transducer Inner Ring Diameters (mm): {inner_diams_mm}")
        print(f"Transducer Outer Ring Diameters (mm): {outer_diams_mm}")

    def _validate_steering(self, tx_params: dict) -> None:
        """
        Validates the transducer steering parameter.
        
        Args:
            tx_params (dict): Raw transducer parameters loaded from yaml file.
        
        Raises:
            ValueError: If steering or any of its subcomponents are missing or not valid type.
                        If steering limits do not make sense physically.
        
        Sets:
            self.xsteering_limits (list): [min_steering_limit, max_steering_limit]
            self.ysteering_limits (list): [min_steering_limit, max_steering_limit]
            self.zsteering_limits (list): [min_steering_limit, max_steering_limit]
        """
        
        if not self.is_steerable:
            return
        
        tx_steering = self._get_param('steering', dict, tx_params)
        tx_xsteering = tx_ysteering = tx_zsteering = None
        
        # Only validate axes that the geometry actually supports
        if 'x' in self.steering_axes:
            tx_xsteering = self._get_param('x', list, tx_steering, parent_key='steering', list_type=(int,float))
            self._validate_limits(tx_xsteering,"steering.x")
            print(f"Transducer X Steering Limits (m): {tx_xsteering}")
        if 'y' in self.steering_axes:
            tx_ysteering = self._get_param('y', list, tx_steering, parent_key='steering', list_type=(int,float))
            self._validate_limits(tx_ysteering,"steering.y")
            print(f"Transducer Y Steering Limits (m): {tx_ysteering}")
        if 'z' in self.steering_axes:
            tx_zsteering = self._get_param('z', list, tx_steering, parent_key='steering', list_type=(int,float))
            self._validate_limits(tx_zsteering,"steering.z")
            print(f"Transducer Z Steering Limits (m): {tx_zsteering}")
        
        self._validate_numeric_list_dict(tx_steering,2,'steering')
        
        # Check negative z steering does not exceed focal length as this is not physically possible
        if 'z' in self.steering_axes:
            abs_zsteering_min = abs(tx_zsteering[0])
            if abs_zsteering_min > self.focal_length:
                raise YAMLParameterError(
                    f"Z minimum steering limit ({abs_zsteering_min}) exceeds focal length distance ({self.focal_length})",
                    full_key='steering.z[0]'
                )
        
        self.xsteering_limits = tx_xsteering
        self.ysteering_limits = tx_ysteering
        self.zsteering_limits = tx_zsteering
        
    def _validate_PlanTUS(self, tx_params: dict) -> None:
        """
        Validates the transducer PlanTUS parameter.
        
        Args:
            tx_params (dict): Raw transducer parameters loaded from yaml file.
        
        Raises:
            ValueError: If listed frequencies do not match transducer frequencies. If FocalDistanceList or 
                        FHMLList are missing from file or are wrong type. If the number of elements between
                        FocalDistanceList and FHMLList do not match.
        
        Sets:
            self.PlanTUS (dict): Validated PlanTUS dict
        """
        
        # PlanTUS is optional; skip validation entirely if the key is absent
        tx_planTUS = self._get_param('PlanTUS', dict, tx_params, optional=True)
        tx_planTUS_new = {}

        # Work from a copy so we can remove matched frequencies and detect any omissions at the end
        tx_freqs = self.frequencies.copy()

        if tx_planTUS is not None:
            print("PlanTUS Parameters")

            for planTUS_key, planTUS_value in tx_planTUS.items():
                
                # Validate PlanTUS frequency
                if not isinstance(planTUS_key,(int,float)):
                    raise YAMLParameterError(
                        f"Frequencies under PlanTUS should be int or float, you put {planTUS_key} ({type(planTUS_key)})",
                        full_key=f'PlanTUS.{planTUS_key}'
                    )
                
                planTUS_freq = int(planTUS_key)
                print(f"    {planTUS_freq} Hz")
                
                # PlanTUS entries must correspond to a frequency already declared for this transducer
                if planTUS_freq not in tx_freqs:
                    raise YAMLParameterError(
                        f"PlanTUS frequency ({planTUS_freq} Hz) is not listed as one of the transducer frequencies",
                        full_key=f'PlanTUS.{planTUS_key}'
                    )
                
                # Validate elements in PlanTUS frequency
                if planTUS_value is None:
                    raise YAMLParameterError(
                        f"FocalDistanceList and FHMLList are missing from {planTUS_key} parameter under PlanTUS parameter",
                        full_key=f'PlanTUS.{planTUS_key}'
                    )
                tx_planTUS_focal_dists = self._get_param('FocalDistanceList', list, planTUS_value, list_type=(int,float), parent_key=f'PlanTUS.{planTUS_key}')
                tx_planTUS_focal_FHMLs = self._get_param('FHMLList', list, planTUS_value, list_type=(int,float), parent_key=f'PlanTUS.{planTUS_key}')
                self._validate_numeric_list_dict(planTUS_value,None,f"PlanTUS {planTUS_key}",allow_negative=False)
                
                # Element Num check
                if len(tx_planTUS_focal_dists) != len(tx_planTUS_focal_FHMLs):
                    raise ValueError(f"Number of elements in FocalDistanceList ({len(tx_planTUS_focal_dists)}) does not match number in FHMLList ({len(tx_planTUS_focal_FHMLs)})")
                                
                # Rename keys
                print("        focal_distances (m):")
                print(f"           {tx_planTUS_focal_dists}")
                print("        FHMLs:")
                print(f"           {tx_planTUS_focal_FHMLs}")
                tx_planTUS_new[planTUS_freq] = {'focal_distances': tx_planTUS_focal_dists, 
                                                'FHMLs': tx_planTUS_focal_FHMLs}
                
                # Remove current PlanTUS freq from check list so we can detect missing entries below
                tx_freqs.remove(planTUS_freq)
            
            # Any frequencies still in tx_freqs that were never covered by a PlanTUS entry
            if len(tx_freqs) > 0:
                missing_details = ", ".join(f"{freq} Hz" for freq in tx_freqs)
                raise ValueError(f"PlanTUS parameter is missing details for following frequencies: {missing_details}")
        else:
            for freq in tx_freqs:
                tx_planTUS_focal_dists_initial = []

                # Determine number of points to measure
                if self.zsteering_limits:
                    start = (self.focal_length + self.zsteering_limits[0]) * 1e3
                    stop = (self.focal_length + self.zsteering_limits[1]) * 1e3
                    tx_planTUS_focal_dists_initial = range(int(start),int(stop),2)
                else:
                    tx_planTUS_focal_dists_initial = [self.focal_length]

                tx_planTUS_new[freq] = {}
                tx_planTUS_new[freq]['FocalDistanceListInitial'] = list(tx_planTUS_focal_dists_initial)
                tx_planTUS_new[freq]['FocalDistanceList'] = []
                tx_planTUS_new[freq]['FHMLs'] = []

        self.PlanTUS = tx_planTUS_new

    # =============================================================================
    # TX FILE CREATION
    # =============================================================================
       
    def _create_tx_files(self) -> None:
        print(f"Creating {self.name} transducer files")
        
        
        # Environment for jinja files
        env = Environment(loader=FileSystemLoader(resource_path(__file__) / ".." / "babel_transducers" / "transducer_templates"), trim_blocks=True, lstrip_blocks=True)
        self.env = env
        
        self._set_tx_file_paths()
        self._create_tx_folder()
        self._create_tx_gui_file()
        self._create_tx_main_file()
        self._create_tx_integration_file()
    
    def _set_tx_file_paths(self):
        tx_parent_folder = CUSTOM_TRANSDUCERS_FOLDER
        tx_folder = tx_parent_folder / f"babel_{self.class_name}"
        tx_default_yaml = tx_folder / "default.yaml"
        tx_main_file = tx_folder / f"babel_{self.class_name}.py"
        tx_form_file = tx_folder / f"{self.class_name}_form.py"
        tx_integration_file = tx_folder / f"babel_integration_{self.class_name}.py"
        
        # Overwrite existing files dialog
        if os.path.exists(tx_folder):
            msgbox = overwrite_msgbox(self.class_name)
            
            if msgbox.exec() == QMessageBox.Yes:
                # Create temp directory to store current version of tx files as
                # backup in case user rejects new transducer design
                self.old_tx_temp_dir = tempfile.mkdtemp()
                shutil.copytree(
                    str(tx_folder),
                    self.old_tx_temp_dir,
                    dirs_exist_ok=True,
                )
                logging.info(f"Existing transducer files copied to temp folder {self.old_tx_temp_dir}")
            else:
                raise ValueError("Cancel Action: Transducer already exists")
        
        self.tx_parent_folder = tx_parent_folder
        self.tx_folder = tx_folder
        self.tx_default_yaml = tx_default_yaml
        self.tx_main_file = tx_main_file
        self.tx_form_file = tx_form_file
        self.tx_integration_file = tx_integration_file
    
    def _create_tx_folder(self) -> None:
        
        # Define the transducers folder path if not already created
        if not os.path.exists(self.tx_parent_folder):
            # Create the directory safely
            self.tx_parent_folder.mkdir(parents=True, exist_ok=True)

        # Delete existing files
        if self.tx_folder.exists():
            shutil.rmtree(self.tx_folder)

        # Create the directory safely
        self.tx_folder.mkdir(parents=True, exist_ok=True)
        
    def _create_tx_main_file(self):
        tx_main_file_template = self.env.get_template("babel_tx.py.jinja")
        
        # Argument formating
        transducer_template = self.geometry_type + "_tx"
        transducer_template_class_name = get_class_name(transducer_template)
        transducer_config = self._format_transducer_config()
        
        # Create Tx Form Text
        tx_main_file_output = tx_main_file_template.render(
            babelbrain_version=self.bb_version,
            template_version=self.template_version,
            transducer_template= "babel_" +transducer_template,
            transducer_template_class_name=transducer_template_class_name,
            transducer_class_name=self.class_name,
            default_yaml=self.tx_default_yaml
        )
        
        # Create Tx Main File
        with open(self.tx_main_file, "w") as f:
            f.write(tx_main_file_output)
            
        # Create default.yaml File
        safe_transducer_config = self._make_yaml_safe(transducer_config)

        with open(self.tx_default_yaml, "w") as f:
            yaml.safe_dump(
                safe_transducer_config,
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        
    def _create_tx_gui_file(self):
        tx_form_template = self.env.get_template("tx_form.py.jinja")

        # Argument formating
        if len(self.steering_axes) == 3:
            xy_mech = "(-10.0, 10.0)"
            multifocal = True
            refocusing = True
            
        else:
            xy_mech = "(-5.0, 5.0)"
            multifocal = False
            refocusing = False
        
        skin_distance = None
        z_mechanic = None
        device_skin_to_target_label = False
        alternative_tissue_warning_value = None
        if self.geometry_type == "flat_array_2D":
            skin_distance = "(-90.0, 90.0)"
            alternative_tissue_warning_value = r'"Tissue layers\nwill be removed!"'
            device_skin_to_target_label = True
        elif self.geometry_type == "flat_annular_array":
            skin_distance = "(-35.0, 0.0)"
            device_skin_to_target_label = True
        elif self.geometry_type == "focused_annular_array":
            skin_distance = "(-50.0, 50.0)"
        elif self.geometry_type == "simple_focused":
            skin_distance = "(-25.0, 5.0)"
        else:
            z_mechanic = "(-90.0, 90.0)"
            alternative_tissue_warning_value = "None"
            
        if "z" in self.steering_axes:
            if self.geometry_type in ["flat_array_2D","focused_array"]:
                steering_z_name = "ZSteering"
            else:
                steering_z_name = "TPODistance"
        else:
            steering_z_name = None
        
        # Create Tx Form Text
        tx_form_output = tx_form_template.render(
            babelbrain_version=self.bb_version,
            template_version=self.template_version,
            tx_name=self.class_name,
            focal_length_adjustable=self.geometry_type == "simple_focused",
            diameter_adjustable=self.geometry_type == "simple_focused",
            multifocal=multifocal,
            refocusing=refocusing,
            distance_outplane_to_focus=self.geometry_type == "simple_focused",
            distance_cone_to_focus=self.geometry_type == "focused_array",
            steering_x="x" in self.steering_axes,
            steering_y="y" in self.steering_axes,
            steering_z="z" in self.steering_axes,
            steering_z_name=steering_z_name,
            device_skin_to_target_label=device_skin_to_target_label,
            xy_mech=xy_mech,
            skin_distance=skin_distance,
            z_mechanic=z_mechanic,
            alternative_tissue_warning_value=alternative_tissue_warning_value
        )
        
        # Create Tx Form File
        with open(self.tx_form_file, "w") as f:
            f.write(tx_form_output)
    
    def _create_tx_integration_file(self):
        tx_integration_file_template = self.env.get_template("babel_integration_tx.py.jinja")
        
        # Argument formating
        transducer_integration_template = "babel_integration_" + self.geometry_type
        
        # Create Tx Form Text
        tx_integration_output = tx_integration_file_template.render(
            babelbrain_version=self.bb_version,
            template_version=self.template_version,
            transducer_integration_template=transducer_integration_template,
        )
        
        # Create Tx Main File
        with open(self.tx_integration_file, "w") as f:
            f.write(tx_integration_output)
    
    def _format_transducer_config(self):
        transducer_config = vars(self).copy()
        del transducer_config['env']
        
        # Important folder paths
        transducer_config["tx_parent_folder"] = str(self.tx_parent_folder.resolve())
        transducer_config["tx_folder"] = str(self.tx_folder.resolve())
        transducer_config["tx_default_yaml"] = str(self.tx_default_yaml.resolve())
        transducer_config["tx_main_file"] = str(self.tx_main_file.resolve())
        transducer_config["tx_form_file"] = str(self.tx_form_file.resolve())
        transducer_config["tx_integration_file"] = str(self.tx_integration_file.resolve())
        
        # Variable renaming
        transducer_config['USFrequencies'] = transducer_config.pop('frequencies')
        transducer_config['NaturalOutPlaneDistance'] = transducer_config.pop('distance_outplane')
        transducer_config['TxDiam'] = transducer_config.pop('aperture_size')
        if self.geometry_type in ["focused_annular_array","flat_annular_array","focused_array"]:
            transducer_config['FocalLength'] = transducer_config.pop('focal_length')
            
            if self.is_annular:
                transducer_config['InDiameters'] = transducer_config['rings']['inner_diameters']
                transducer_config['OutDiameters'] = transducer_config['rings']['outer_diameters']
                transducer_config.pop('rings')
        
        if "x" in self.steering_axes:
            transducer_config['MinimalXSteering'] = transducer_config['xsteering_limits'][0]
            transducer_config['MaximalXSteering'] = transducer_config['xsteering_limits'][-1]
            
        if "y" in self.steering_axes:
            transducer_config['MinimalYSteering'] = transducer_config['ysteering_limits'][0]
            transducer_config['MaximalYSteering'] = transducer_config['ysteering_limits'][-1]
            
        if "z" in self.steering_axes:
            transducer_config['MinimalZSteering'] = transducer_config['zsteering_limits'][0]
            transducer_config['MaximalZSteering'] = transducer_config['zsteering_limits'][-1]
            transducer_config['DefaultZSteering'] = float(np.sum(transducer_config['zsteering_limits'])/len(transducer_config['zsteering_limits']))
        
        # Added Default variable values
        if self.geometry_type == 'simple_focused':
            transducer_config['MaxDistanceToSkin'] = 50 # mm
            transducer_config['MaxNegativeDistance'] = 10   # mm
        elif self.geometry_type in ['focused_annular_array','flat_annular_array']:
            transducer_config['MaxDistanceToSkin'] = 50 # mm
            transducer_config['MaxNegativeDistance'] = 10   # mm
            transducer_config['MinimalTPODistance'] = 8.0e-3   # m
            transducer_config['MaximalTPODistance'] = 120.0e-3  # m
        elif self.geometry_type in ['flat_array_2D']:
            transducer_config['MaxDistanceToSkin'] = 50 # mm
            transducer_config['MaxNegativeDistance'] = 10   # mm
        elif self.geometry_type in ['focused_array']:
            transducer_config['MinimalDistanceConeToFocus'] = 10.0e-3 # m
            transducer_config['MaximalDistanceConeToFocus'] = 129.0e-3 # m
            transducer_config['DefaultDistanceConeToFocus'] = (transducer_config['MinimalDistanceConeToFocus'] + transducer_config['MaximalDistanceConeToFocus']) / 2
            
        return transducer_config
    
    def _make_yaml_safe(self, value):
        if isinstance(value, dict):
            return {self._make_yaml_safe(k): self._make_yaml_safe(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._make_yaml_safe(v) for v in value]

        if isinstance(value, np.ndarray):
            return [self._make_yaml_safe(v) for v in value.tolist()]

        if isinstance(value, np.generic):
            return value.item()

        return value

    # =============================================================================
    # TRANSDUCER VALIDATION
    # =============================================================================
    
    def _validate_tx(self):

        sys.path.insert(0, str(CUSTOM_TRANSDUCERS_FOLDER))
        self.TxIntegration = importlib.import_module(f"babel_{self.class_name}.babel_integration_{self.class_name}")

        # Acoustic Water Sims for PlanTUS
        if not self.PlanTUS[self.frequencies[0]]['FHMLs']:
            if self.geometry_type == 'flat_annular_array':
                # FHML calculation is wonky for this transducer type
                pass 
            else:
                for freq in self.PlanTUS:
                    focal_dists_per_freq, FHMLs_per_freq = self._run_rayleigh_PlanTUS(freq,normalized_pressure=False,plot_FHML=False)
                    self.PlanTUS[freq]['FocalDistanceList'] = focal_dists_per_freq
                    self.PlanTUS[freq]['FHMLs'] = FHMLs_per_freq
                    del self.PlanTUS[freq]['FocalDistanceListInitial']
                    
                # Update default file
                with open(self.tx_default_yaml, "r") as f:
                    data = yaml.safe_load(f)

                data["PlanTUS"]= self.PlanTUS

                with open(self.tx_default_yaml, "w") as f:
                    yaml.safe_dump(
                        self._make_yaml_safe(data),
                        f,
                        default_flow_style=False,
                        sort_keys=False,
                    )

        # Acoustics Water Sim
        tx_data, acoustics_water_plot, grid_info = self._run_rayleigh()
        acoustics_water_plot = np.abs(acoustics_water_plot)
    
        user_verification_dialog = TransducerVerificationDialog(
            tx_data=tx_data,
            acoustic_data=acoustics_water_plot,
            grid_info=grid_info,
            parent=None,
        )

        if user_verification_dialog.exec() == QDialog.DialogCode.Accepted:
            print("User approved the generated transducer.")
        else:
            if self.old_tx_temp_dir:
                logging.info('User did not approve of new transducer design, restoring previous version')
                shutil.copytree(
                    self.old_tx_temp_dir,
                    str(self.tx_folder),
                    dirs_exist_ok=True,
                )
                
                msgbox = restore_msgbox()
                msgbox.exec()
            
            raise ValueError("Cancel Action: User did not approve transducer design")
        
    
    def _run_rayleigh(self):
        
        args = {}
        args['Aperture'] = self.aperture_size
        args['Frequency'] = self.frequencies[0]
        args['FocalLength'] = self.focal_length
        if 'x' in self.steering_axes:   
            args['XSteering'] = 0.0
        if 'y' in self.steering_axes:   
            args['YSteering'] = 0.0
        if 'z' in self.steering_axes:   
            args['ZSteering'] = 0.0
        if len(self.steering_axes) == 3:
            args['RotationZ'] = 0.0
        if self.geometry_type in ['focused_array']:
            args['DistanceConeToFocus'] = self.focal_length - self.distance_outplane
            args['coordinate_system'] = self.coordinate_system
        if self.geometry_type in ['focused_array','flat_array_2D']:
            args['elements'] = self.elements
            args['num_elements'] = self.num_elements
            args['element_size'] = self.element_size
        if self.is_annular:
            args['InDiameters'] = np.array(self.rings['inner_diameters'])
            args['OutDiameters'] = np.array(self.rings['outer_diameters'])
            
        sim_conditions = self.TxIntegration.SimulationConditions(**args)
        
        if self.geometry_type in ['focused_array']:
            sim_conditions.GenTransducerGeom()
        elif self.geometry_type in ['flat_array_2D']:
            sim_conditions._Tx = sim_conditions.GenTransducerGeom()
        else:
            sim_conditions._Tx = sim_conditions.GenTx()

        Material = {}
        Material['Water']=     np.array([1000.0, SpeedofSoundWater(20.0), 0.0   ,   0.0,                   0.0] )
        cwvnb_extlay=np.array(2*np.pi*sim_conditions._Frequency/(Material['Water'][1])+1j*0).astype(np.complex64)
        
        #Limits of domain, in m
        radius = self.aperture_size/2*1.5
        depth = self.focal_length*2
        xfmin=-radius
        xfmax=radius
        yfmin=-radius
        yfmax=radius
        zfmin=0
        zfmax=max(depth,xfmax-xfmin)
        
        spatial_step = SpeedofSoundWater(20.0) / self.frequencies[0] / 6

        xfield = np.linspace(xfmin,xfmax,int(np.ceil((xfmax-xfmin)/spatial_step)+1))
        yfield = np.linspace(yfmin,yfmax,int(np.ceil((yfmax-yfmin)/spatial_step)+1))
        zfield = np.linspace(zfmin,zfmax,int(np.ceil((zfmax-zfmin)/spatial_step)+1))
        nxf=len(xfield)
        nyf=len(yfield)
        nzf=len(zfield)
        xp,yp,zp=np.meshgrid(xfield,yfield,zfield)

        Amp = 60e3/Material['Water'][0]/SpeedofSoundWater(20.0) #60 kPa

        sim_conditions._SourceAmpPa = Amp
        u0=(np.ones((sim_conditions._Tx['center'].shape[0],1),np.float32)+ 1j*np.zeros((sim_conditions._Tx['center'].shape[0],1),np.float32))*sim_conditions._SourceAmpPa
        
        rf=np.hstack(
            (
                np.reshape(xp,(nxf*nyf*nzf,1)),
                np.reshape(yp,(nxf*nyf*nzf,1)),
                np.reshape(zp,(nxf*nyf*nzf,1))
            )
        ).astype(np.float32)
        
        u2 = self._run_forward_simple(
            cwvnb_extlay,
            sim_conditions._Tx['center'].astype(np.float32),
            sim_conditions._Tx['ds'].astype(np.float32),
            u0,
            rf
        )
        u2 *= Material['Water'][0]*Material['Water'][1]
        u2 = np.reshape(u2,xp.shape)
        
        grid_info = {}
        grid_info['xfmin'] = xfmin
        grid_info['xfmax'] = xfmax
        grid_info['yfmin'] = yfmin
        grid_info['yfmax'] = yfmax
        grid_info['zfmin'] = zfmin
        grid_info['zfmax'] = zfmax
        grid_info['spatial_step'] = spatial_step
        
        return sim_conditions._Tx, u2.T, grid_info

    def _make_legend_interactive(self, fig, ax, plot_lines, plot_data):
        legend = ax.legend()
        legend_lines = legend.get_lines()
        legend_texts = legend.get_texts()

        selected_line = {'index': None}

        # FHML markers, hidden until a single line is selected
        lower_line = ax.axvline(
            x=0,
            linestyle='--',
            color='blue',
            label='Lower',
            visible=False
        )
        upper_line = ax.axvline(
            x=0,
            linestyle='--',
            color='blue',
            label='Upper',
            visible=False
        )
        half_max_line = ax.axhline(
            y=0,
            linestyle='--',
            color='orange',
            label='Half Max',
            visible=False
        )

        for legend_line, legend_text in zip(legend_lines, legend_texts):
            legend_line.set_picker(True)
            legend_line.set_pickradius(5)
            legend_text.set_picker(True)

        def on_pick(event):
            if event.artist in legend_lines:
                selected_idx = legend_lines.index(event.artist)
            elif event.artist in legend_texts:
                selected_idx = legend_texts.index(event.artist)
            else:
                return

            # Clicking the selected line again restores all lines
            if selected_line['index'] == selected_idx:
                selected_line['index'] = None

                for line, legend_line, legend_text in zip(
                    plot_lines,
                    legend_lines,
                    legend_texts
                ):
                    line.set_visible(True)
                    legend_line.set_alpha(1.0)
                    legend_text.set_alpha(1.0)

                lower_line.set_visible(False)
                upper_line.set_visible(False)
                half_max_line.set_visible(False)

            # Otherwise only show the selected line
            else:
                selected_line['index'] = selected_idx

                for i, (line, legend_line, legend_text) in enumerate(zip(
                    plot_lines,
                    legend_lines,
                    legend_texts
                )):
                    visible = i == selected_idx
                    line.set_visible(visible)
                    legend_line.set_alpha(1.0 if visible else 0.3)
                    legend_text.set_alpha(1.0 if visible else 0.3)

                data = plot_data[selected_idx]

                lower_line.set_xdata([data['left'], data['left']])
                upper_line.set_xdata([data['right'], data['right']])
                half_max_line.set_ydata([data['half_max'], data['half_max']])

                lower_line.set_visible(True)
                upper_line.set_visible(True)
                half_max_line.set_visible(True)

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('pick_event', on_pick)

    def _run_rayleigh_PlanTUS(self,freq, normalized_pressure=True, plot_FHML=False):
            
        args = {}
        args['Aperture'] = self.aperture_size
        args['Frequency'] = freq
        args['FocalLength'] = self.focal_length
        if 'x' in self.steering_axes:
            args['XSteering'] = 0.0
        if 'y' in self.steering_axes:
            args['YSteering'] = 0.0
        if 'z' in self.steering_axes:
            args['ZSteering'] = 0.0 # focal_dist/1e3 - self.focal_length
        if len(self.steering_axes) == 3:
            args['RotationZ'] = 0.0
        if self.geometry_type in ['focused_array']:
            args['DistanceConeToFocus'] = self.focal_length # - self.distance_outplane
            args['coordinate_system'] = self.coordinate_system
        if self.geometry_type in ['focused_array','flat_array_2D']:
            args['elements'] = self.elements
            args['num_elements'] = self.num_elements
            args['element_size'] = self.element_size
        if self.is_annular:
            args['InDiameters'] = np.array(self.rings['inner_diameters'])
            args['OutDiameters'] = np.array(self.rings['outer_diameters'])

        sim_conditions = self.TxIntegration.SimulationConditions(**args)

        if self.geometry_type in ['focused_array']:
            sim_conditions.GenTransducerGeom()
        elif self.geometry_type in ['flat_array_2D']:
            sim_conditions._Tx = sim_conditions.GenTransducerGeom()
        else:
            sim_conditions._Tx = sim_conditions.GenTx()
        print('Tx z  min',sim_conditions._Tx['center'][:,2].min())
        
        Material = {}
        Material['Water'] = np.array([1000.0, SpeedofSoundWater(20.0), 0.0, 0.0, 0.0] )
        sim_conditions._Material = Material
        
        cwvnb_extlay = np.array(
            2*np.pi*sim_conditions._Frequency/Material['Water'][1]+1j*0
        ).astype(np.complex64)
        
        spatial_step = SpeedofSoundWater(20.0) / sim_conditions._Frequency / 6
        sim_conditions._SpatialStep = spatial_step
        
        #Limits of domain, in m
        radius = self.aperture_size/2*1.5
        depth = self.focal_length*2
        xfmin=-radius
        xfmax=radius
        yfmin=-radius
        yfmax=radius
        zfmin=0
        zfmax=max(depth,xfmax-xfmin)
        xfield = np.linspace(xfmin, xfmax,int(np.ceil((xfmax - xfmin) / spatial_step) + 1) | 1)
        yfield = np.linspace(yfmin, yfmax,int(np.ceil((yfmax - yfmin) / spatial_step) + 1) | 1)
        zfield = np.linspace(zfmin, zfmax,int(np.ceil((zfmax - zfmin) / spatial_step) + 1) | 1)
        sim_conditions._XDim = xfield
        sim_conditions._YDim = yfield
        sim_conditions._ZDim = zfield
        
        amp = 60e3/Material['Water'][0]/SpeedofSoundWater(20.0) #60 kPa
        
        sim_conditions._ZSourceLocation = 0

        if self.geometry_type == 'simple_focused':
            n_elems = 1
        else:
            n_elems = sim_conditions._Tx['NumberElems']
        n_total = sim_conditions._Tx['center'].shape[0]

        focal_dists = []
        FHMLs = []

        if plot_FHML:
            fig = Figure(tight_layout=True)
            canvas = FigureCanvasQTAgg(fig)
            ax = fig.add_subplot(111)
            plot_lines = []
            plot_data = []

        for focal_idx, focal_dist in enumerate(
            self.PlanTUS[freq]['FocalDistanceListInitial']
        ):
            new_zsteering = focal_dist/1e3 - self.focal_length
            new_target = self.focal_length+new_zsteering
            
            sim_conditions._FocalSpotLocation = np.array([
                len(xfield)//2,
                len(yfield)//2,
                np.argmin(abs(sim_conditions._ZDim-new_target))
            ])
            
            # ------------------------------------------------------------------ #
            #  Steering / phase computation                                      #
            # ------------------------------------------------------------------ #
            if self.geometry_type in ['flat_annular_array','focused_annular_array']:
                # Forward-propagate each element's sub-panels to the focal point and
                # compute the conjugate phase needed to steer to that point.
                center = np.zeros((1, 3), np.float32)
                center[0,0] = sim_conditions._XDim[sim_conditions._FocalSpotLocation[0]]
                center[0,1] = sim_conditions._YDim[sim_conditions._FocalSpotLocation[1]]
                center[0,2] = sim_conditions._ZDim[sim_conditions._FocalSpotLocation[2]]
                print('center', center)
                print('Z location', sim_conditions._ZDim[sim_conditions._ZSourceLocation])

                u2back = np.zeros(n_elems, np.complex64)
                nBase = 0
                print('Locations Tx and center', sim_conditions._Tx['center'].min(axis=0), center)
                for n in range(n_elems):
                    n_sub = sim_conditions._Tx['elemdims'][n][0]
                    u0_sub = np.ones(n_sub, np.complex64)
                    SelCenters = sim_conditions._Tx['center'][nBase:nBase+n_sub, :].astype(np.float32)
                    SelDs      = sim_conditions._Tx['ds'][nBase:nBase+n_sub, :].astype(np.float32)
                    u2back[n] = self._run_forward_simple(
                        cwvnb_extlay,
                        SelCenters,
                        SelDs,
                        u0_sub,
                        center
                    )[0]
                    nBase += n_sub

                AllPhi = np.zeros(n_elems)
                for n in range(n_elems):
                    AllPhi[n] = -np.angle(u2back[n])

                print('Phase for array: [', np.rad2deg(AllPhi).tolist(), ']')

                u0 = np.zeros((n_total, 1), np.complex64)
                nBase = 0
                for n in range(n_elems):
                    n_sub = sim_conditions._Tx['elemdims'][n][0]
                    u0[nBase:nBase+n_sub] = (amp * np.exp(1j*AllPhi[n])).astype(np.complex64)
                    nBase += n_sub

            elif self.geometry_type in ['focused_array','flat_array_2D']:
                if not np.isclose(new_target, self.focal_length):
                    # Propagate from the focal point to each element centre (inverse
                    # direction), then conjugate to obtain the steering phase.
                    ds = np.ones((1)) * sim_conditions._SpatialStep**2
                    u0_focal = np.zeros((1), np.complex64)
                    u0_focal[0] = 1+0j
                    center = np.zeros((1, 3), np.float32)
                    center[0,0] = sim_conditions._XDim[sim_conditions._FocalSpotLocation[0]]
                    center[0,1] = sim_conditions._YDim[sim_conditions._FocalSpotLocation[1]]
                    center[0,2] = sim_conditions._ZDim[sim_conditions._FocalSpotLocation[2]]
                    print('center', center)

                    u2back = self._run_forward_simple(
                        cwvnb_extlay,
                        center,
                        ds.astype(np.float32),
                        u0_focal,
                        sim_conditions._Tx['elemcenter'].astype(np.float32)
                    )
                    u0 = np.zeros((n_total, 1), np.complex64)
                    nBase = 0
                    for n in range(n_elems):
                        phi = np.angle(np.conjugate(u2back[n]))
                        u0[nBase:nBase+sim_conditions._Tx['elemdims']] = (amp * np.exp(1j*phi)).astype(np.complex64)
                        nBase += sim_conditions._Tx['elemdims']
                else:
                    u0 = (np.ones((n_total, 1), np.float32)
                        + 1j*np.zeros((n_total, 1), np.float32)) * amp

            elif self.geometry_type == 'simple_focused':  # 'none'
                u0 = (np.ones((n_total, 1), np.float32)
                    + 1j*np.zeros((n_total, 1), np.float32)) * amp

            # ------------------------------------------------------------------ #
            #  Perform Forward Rayleigh                                          #
            # ------------------------------------------------------------------ #
            rf = np.column_stack((
                np.zeros_like(zfield),
                np.zeros_like(zfield),
                zfield
            )).astype(np.float32)

            u2 = self._run_forward_simple(
                cwvnb_extlay,
                sim_conditions._Tx['center'].astype(np.float32),
                sim_conditions._Tx['ds'].astype(np.float32),
                u0,
                rf,
            )
            
            u2_1D = abs(u2)
            u2_1D *= Material['Water'][0]*Material['Water'][1] # Convert to pressure
            if normalized_pressure:
                u2_1D /= max(u2_1D)

            # ------------------------------------------------------------------ #
            #  Calculate/Store FHML                                              #
            # ------------------------------------------------------------------ #
            peak_idx = np.argmax(u2_1D)
            half_max = u2_1D[peak_idx] / 2

            # Find points above half maximum
            above = u2_1D >= half_max

            # Find contiguous region containing the peak
            left = peak_idx
            while left > 0 and above[left - 1]:
                left -= 1

            right = peak_idx
            while right < len(u2_1D) - 1 and above[right + 1]:
                right += 1

            FHML = (zfield[right] - zfield[left])*1e3
            focal_dist = zfield[peak_idx]*1e3
            
            focal_dists.append(np.round(focal_dist,2))
            FHMLs.append(np.round(FHML,2))
            
            # ------------------------------------------------------------------ #
            #  Plot FHML                                                         #
            # ------------------------------------------------------------------ #
            if plot_FHML and (focal_idx % 4 == 0 or np.isclose(new_target, self.focal_length)): # Capture every 4 line plots as well as focal point plot
                line, = ax.plot(
                    zfield*1e3,
                    u2_1D,
                    label=f'{focal_dist:.2f} mm'
                )
                plot_lines.append(line)

                plot_data.append({
                    'left': zfield[left]*1e3,
                    'right': zfield[right]*1e3,
                    'half_max': half_max,
                })

        if plot_FHML:
            ax.set_xlabel('Axial Distance (mm)')
            if normalized_pressure:
                ax.set_ylabel('Normalized Pressure')
            else:
                ax.set_ylabel('Pressure (Pa)')
            ax.set_title(f'Rayleigh Pressure Profiles - {freq/1e3:g} kHz')

            if plot_lines:
                self._make_legend_interactive(
                    fig,
                    ax,
                    plot_lines,
                    plot_data
                )

            dialog = QDialog()
            dialog.setWindowTitle(f'Rayleigh Pressure Profiles - {freq/1e3:g} kHz')
            dialog.resize(800, 500)
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(PlotWidget(canvas))
            canvas.draw()
            dialog.exec()

        return focal_dists, FHMLs

    def _initialize_gpu(self):
        if not self.is_gpu_initialized:
            if self.computing_backend=='CUDA':
                InitCuda(self.gpu)
            elif self.computing_backend=='OpenCL':
                InitOpenCL(self.gpu)
            elif self.computing_backend=='Metal':
                InitMetal(self.gpu)
            elif self.computing_backend=='Server':
                pass
            
            self.is_gpu_initialized = True
        else:
            return

    def _run_forward_simple(self,cwvnb_extlay,center,ds,u0,rf):
        
        if self.computing_backend in 'Server':
            remote_calc = RunServerCalculation(
                step=RAYLEIGH_TEST,
                server=self.remote_server,
                standalone_args={
                    'cwvnb_extlay': cwvnb_extlay,
                    'center': center,
                    'ds': ds,
                    'u0': u0,
                    'rf': rf,
                },
            )
            u2 = remote_calc.run()
        else:
            self._initialize_gpu()
            u2 = ForwardSimple(
                cwvnb_extlay,
                center,
                ds,
                u0,
                rf,
                deviceMetal=self.gpu
            )
            
        return u2