import os
import glob
import logging

import pytest

@pytest.mark.step1
@pytest.mark.skip(reason="Placeholder test")
class TestStep1:
    def test_step1_valid_case(self,qtbot,
                              trajectory,
                              transducer,
                              scan_type,
                              dataset,
                              babelbrain_widget,
                              load_files,
                              compare_data,
                              tmp_path):

        pass

    def test_step1_invalid_case(self,qtbot,trajectory,transducer,scan_type,dataset,babelbrain_widget,tmp_path):

        # Run Step 1
        babelbrain_widget.Widget.CalculatePlanningMask.click()

        # Wait for step 1 completion before continuing. Test timeouts after 15 min have past
        qtbot.waitUntil(babelbrain_widget.Widget.tabWidget.isEnabled,timeout=900000)

        # Placeholder test. Need to find a way to grab the specific error generated.
        assert babelbrain_widget.testing_error == True