# Pytest Setup and Usage Guide

This guide provides instructions for setting up and running pytest tests for BabelBrain, along with additional instructions for using it in Visual Studio Code.

## Installation
Before you begin, ensure you have Pytest installed in your environment.

``` bash
pip install pytest
```

The following packages are also required for BabelBrain pytests:
- pytest-html (Generating html test reports)
- pytest-metadata 
- pytest-qt (Handling QT applications)

```bash
pip install pytest-html, pytest-metadata, pytest-qt, pytest-xdist
```

These optional packages may also be installed:
- pytest-xdist (Parallelization of tests to speed up test run time, avoid using for GPU tests)
- pytest-benchmark (Evaluate code performance)
- pytest-profiling (Useful for determing code bottlenecks)
- pytest-sugar (Improve readability of pytest output)

```bash
pip install pytest-benchmark, pytest-profiling, pytest-sugar
```

## Writing Tests

Pytest allows you to write tests using Python's built-in `assert`
statement. Test files should be named with the prefix `test_` or suffix `_test` so that pytest can automatically discover them.

Here's an example of a simple pytest test function:

```python
# test_example.py

def test_addition():
    assert 1 == 2
```

## Running Tests
### In terminal

To run pytest tests, simply execute the `pytest` command in your terminal, pointing it to the directory containing your test files:

```bash
pytest <path_to_tests_directory>
```

Certain tests can be selected with the -k argument as shown in the example below:

```bash
pytest <path_to_tests_directory> -k "test_addition"
```

A full list of pytest arguments can be found [here](https://docs.pytest.org/en/6.2.x/usage.html).

Note that a configuration has already been set up in the `pytest.ini` file and will be called automatically.

### In Visual Studio Code

Open your project in Visual Studio Code. Install Python and Test Explorer UI extensions for Visual Studio Code.

Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P on macOS) and select "Open Workspace Settings (JSON)". Then set up the following parameters:

```json
{
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "Tests",
    ],
    "python.testing.pytestPath": "<path_to_your_pytest>"
}
```
Example: For a conda environment, the pytest path looks something like 

`<user>/miniconda3/envs/<unit_testing_env>/Lib/site-packages/pytest`

Visual Studio Code will now discover pytest tests and display them in the testing tab; You can run them using the UI. 

If they do not appear, open the Command Palette again and select "Python: Select Interpreter" and choose the one in the same environment as your pytest package.

Note that the commandline arguments are still specified in the `pytest.ini` file and are run automatically.

Alternatively, tests can be run from the integrated terminal similar to previous section.


### Additional Setup for BabelBrain testing
BabelBrain tests require a file path be specified for the test data, certain tests also require the GPU device name. To do this, create a `config.ini` file in Tests folder (i.e. same level as pytest.ini)
<pre>
BabelBrain/
├── Tests/
│     ├── Integration/
│     │     ├── integration_test_1.py
│     │     ├── integration_test_2.py
│     │     └── ...
│     ├── Unit/
│     │     ├── unit_test_1.py
│     │     ├── unit_test_2.py
│     │     └── ...
│     ├── <b>config.ini</b>
│     ├── conftest.py
│     ├── pytest.ini
│     └── ...
└── ...
</pre>

The `config.ini` file should be formatted as 
```plaintext
[Paths]
data_folder_path = <path to test data>

[GPU]
device_name = <GPU device name>
```
Note that your test data folder should have the following structure which is slightly modified from our current [image database](https://zenodo.org/). We are still in the process of generating our truth data, therefore you will need to create your own to run the tests.

<pre>
BabelBrain_Test_Data/
├── Dataset_1/
│     ├── m2m_Dataset_1/
│     │     └── ...
│     ├── Trajectories/
│     │     ├── Target_1.txt
│     │     ├── Target_2.txt
│     │     └── ...
│     ├── Truth/
│     │     ├── Integration/
│     │     │     ├── T1W_Only/
│     │     │     │     ├── Transducer_1/
│     │     │     │     │     ├── Target_1/
│     │     │     │     │     │     ├── Truth_file_1
│     │     │     │     │     │     ├── Truth_file_2
│     │     │     │     │     │     └── ...
│     │     │     │     │     ├── Target_2/
│     │     │     │     │     │     └── ...
│     │     │     │     │     └── ...
│     │     │     │     ├── Transducer_2/
│     │     │     │     │     └── ...
│     │     │     │     └── ...
│     │     │     ├── T1W_with_CT/
│     │     │     │     └── ...
│     │     │     ├── T1W_with_ZTE/
│     │     │     │     └── ...
│     │     │     └── ...
│     │     └── Unit/
│     │           ├── Unit_Test_1/
│     │           │     ├── Truth_file_1
│     │           │     ├── Truth_file_2
│     │           │     └── ...
│     │           ├── Unit_Test_2/
│     │           │     └── ...
│     │           └── ...
│     └── ...
├── Dataset_2/
│     └── ...
├── Thermal_Profiles/
│     ├── Profile_1.yaml
│     ├── Profile_2.yaml
│     └── ...
└── ...
</pre>

The names of datasets, trajectories, and transducers can be modified in the parameters section in the `conftest.py` file.

## Further Reading
For more information about pytest and its features, check out the [pytest documentation](https://docs.pytest.org/en/latest/).

