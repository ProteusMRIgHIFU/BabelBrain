import pytest
import pytest_html

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    
    if report.when == "call":
        extra = getattr(report, "extra", [])

        for screenshot in ["intersection1_screenshot","intersection2_screenshot","intersection3_screenshot","intersection4_screenshot"]:
            if hasattr(item, screenshot):
                extra.append(pytest_html.extras.html('<img src="data:image/png;base64,{}">'.format(getattr(item,screenshot))))
            
        report.extra = extra