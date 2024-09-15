# Examples of basic tests

Create in the `Tests` directory a `config.ini` file with similar entries as shown below

```
[Paths]
data_folder_path = /Users/spichardo/Documents/TempForSim/
[GPU]
device_name = M3
```

## Minimal test per release
The test below should the bare minimum at every release

It will test that no execution errors occurs processing one single dataset with MRI-only, CT and ZTE inputs. It will run 30 simulations in totals, 10 transducers $\times$ 2 datasets.

`pytest Tests -k 'test_steps_normal and brainsight and Deep_Target and ( -CT- or -NONE- ) and ID_0082'`