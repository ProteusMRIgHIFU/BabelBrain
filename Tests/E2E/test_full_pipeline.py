import configparser
import glob
import logging
import os

import pytest

def test_full_pipeline_normal(qtbot,babelbrain_widget,image_to_base64,request,transducer):

    # Save plot screenshot to be added to html report later
    request.node.screenshots = []

    # Start babelbrain instance
    bb_widget = babelbrain_widget()
    
    # Run Step 1
    bb_widget.testing_error = False
    bb_widget.Widget.CalculatePlanningMask.click()

    # Wait for step 1 completion before continuing. Test timeouts after 15 min have past
    qtbot.waitUntil(bb_widget.Widget.tabWidget.isEnabled,timeout=900000)
    qtbot.wait(1000) # Wait for plots to display
    
    # Take screenshot of step 1 results
    screenshot = qtbot.screenshot(bb_widget)
    request.node.screenshots.append(image_to_base64(screenshot))

    # Check if step 1 failed
    if bb_widget.testing_error == True:
        pytest.fail(f"Test failed due to error in execution")

    # Run Step 2
    bb_widget.Widget.tabWidget.setCurrentIndex(1)
    bb_widget.AcSim.Widget.CalculateAcField.click()

    # Wait for step 2 completion before continuing. Test timeouts after 15 min have past
    qtbot.waitUntil(bb_widget.Widget.tabWidget.isEnabled,timeout=900000)
    qtbot.wait(1000) # Wait for plots to display

    # Take screenshot of step 2 results
    screenshot = qtbot.screenshot(bb_widget)
    request.node.screenshots.append(image_to_base64(screenshot))

    if bb_widget.testing_error == True:
        pytest.fail(f"Test failed due to error in execution")

    # Run Step 3
    bb_widget.Widget.tabWidget.setCurrentIndex(2)
    bb_widget.ThermalSim.Widget.CalculateThermal.click()

    # Wait for step 3 completion before continuing. Test timeouts after 15 min have past
    qtbot.waitUntil(bb_widget.Widget.tabWidget.isEnabled,timeout=900000)
    qtbot.wait(1000) # Wait for plots to display

    # Take screenshot of step 3 results
    screenshot = qtbot.screenshot(bb_widget)
    request.node.screenshots.append(image_to_base64(screenshot))

    if bb_widget.testing_error == True:
        pytest.fail(f"Test failed due to error in execution")
    
    with qtbot.captureExceptions() as exceptions:
        # Run Export CSV Command
        bb_widget.ThermalSim.Widget.ExportSummary.click()
        
        # Run Export Maps Command
        bb_widget.ThermalSim.Widget.ExportMaps.click()
        
    if len(exceptions) > 0:
        pytest.fail(f"Test failed due to error in execution: {exceptions}")
    
def test_full_pipeline_regression_normal(qtbot,babelbrain_widget,image_to_base64,compare_BabelBrain_Outputs,request,tmp_path,tolerance,get_config_dirs):
    config_dirs = get_config_dirs
    ref_output_dir = config_dirs['ref_dir_1']
    
    # Save plot screenshot to be added to html report later
    request.node.screenshots = []

    # Start babelbrain instance
    bb_widget = babelbrain_widget()
    
    # Run Step 1
    bb_widget.testing_error = False
    bb_widget.Widget.CalculatePlanningMask.click()

    # Wait for step 1 completion before continuing. Test timeouts after 15 min have past
    qtbot.waitUntil(bb_widget.Widget.tabWidget.isEnabled,timeout=900000)
    qtbot.wait(1000) # Wait for plots to display
    
    # Take screenshot of step 1 results
    screenshot = qtbot.screenshot(bb_widget)
    request.node.screenshots.append(image_to_base64(screenshot))

    # Check if step 1 failed
    if bb_widget.testing_error == True:
        pytest.fail(f"Test failed due to error in execution")

    # Run Step 2
    bb_widget.Widget.tabWidget.setCurrentIndex(1)
    bb_widget.AcSim.Widget.CalculateAcField.click()

    # Wait for step 2 completion before continuing. Test timeouts after 15 min have past
    qtbot.waitUntil(bb_widget.Widget.tabWidget.isEnabled,timeout=900000)
    qtbot.wait(1000) # Wait for plots to display

    # Take screenshot of step 2 results
    screenshot = qtbot.screenshot(bb_widget)
    request.node.screenshots.append(image_to_base64(screenshot))

    if bb_widget.testing_error == True:
        pytest.fail(f"Test failed due to error in execution")

    # Run Step 3
    bb_widget.Widget.tabWidget.setCurrentIndex(2)
    bb_widget.ThermalSim.Widget.CalculateThermal.click()

    # Wait for step 3 completion before continuing. Test timeouts after 15 min have past
    qtbot.waitUntil(bb_widget.Widget.tabWidget.isEnabled,timeout=900000)
    qtbot.wait(1000) # Wait for plots to display

    # Take screenshot of step 3 results
    screenshot = qtbot.screenshot(bb_widget)
    request.node.screenshots.append(image_to_base64(screenshot))

    if bb_widget.testing_error == True:
        pytest.fail(f"Test failed due to error in execution")
    
    with qtbot.captureExceptions() as exceptions:
        # Run Export CSV Command
        bb_widget.ThermalSim.Widget.ExportSummary.click()
        
        # Run Export Maps Command
        bb_widget.ThermalSim.Widget.ExportMaps.click()
        
    if len(exceptions) > 0:
        pytest.fail(f"Test failed due to error in execution: {exceptions}")
        
    # Compare output against reference outputs
    bb_outputs_match = compare_BabelBrain_Outputs(ref_folder = ref_output_dir, test_folder = tmp_path, tolerance=tolerance)
    
    assert bb_outputs_match == True, "There were differences between the h5 files, see logged warnings for more details"
    
def test_full_pipeline_two_outputs(compare_BabelBrain_Outputs,tolerance,get_config_dirs):

    config_dirs = get_config_dirs
    ref_output_dir = config_dirs['ref_dir_1']
    ref_output_dir_2 = config_dirs['ref_dir_2']
    
    bb_outputs_match = compare_BabelBrain_Outputs(ref_folder = ref_output_dir, test_folder = ref_output_dir_2,tolerance=tolerance)
        
    assert bb_outputs_match == True, "There were differences between the h5 files, see logged warnings for more details"