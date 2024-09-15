# Examples of basic tests

## Minimal test per release
The test below should the bare minimum at every release

It will test that no execution errors occurs processing one single dataset with MRI-only and CT inputs. It will run 20 simulations in totals, 10 transducers $\times$ 2 datasets.

`pytest Tests -k 'test_steps_normal and brainsight and Deep_Target and ( -CT- or -NONE- ) and 0p55'`