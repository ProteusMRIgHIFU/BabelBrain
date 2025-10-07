# Examples of basic tests

Create in the `Tests` directory a `config.ini` file with similar entries as shown below

```
[Paths]
data_folder_path = /Users/spichardo/Documents/TempForSim/
[GPU]
device_name = M3
```

## Minimal test per release
The test below should the bare minimum at every release.\
It will test that no execution errors occurs processing one single dataset with MRI-only, CT and ZTE inputs. It will run 33 simulations in totals, 11 transducers $\times$ 3 types of imaging input.

`pytest Tests -k 'test_full_pipeline_normal and brainsight and Deep_Target and ( -CT- or -NONE- or -ZTE- ) and ID_0082'`

if on macOS/Linux.

In Windows, run first a BAT generator for the test with

`pytest getPytestWin.py`

them run 

`runPytestWin.bat`

It will run one test each time as GPU code has tendency to crash in Windows if running too many in a row. It has been difficult to predict