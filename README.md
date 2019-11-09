# HiCRestorationTools
Two command line tools (Neural network and simple interpolation) for restoring missing areas in Hi-C maps

## Usage
Simple interpolation model:

```
python3 neural_network.py PATH_TO_YOUR_COOL_FILE [DEFECT_THRESHOLD]
```
Neural network model:

```
python3 interpolation.py PATH_TO_YOUR_COOL_FILE [DEFECT_THRESHOLD]
```

DEFECT_THRESHOLD is an optional argument which defines minimal percent of non-zero values in row/column for this row/column to be valid.
By default DEFECT_THRESHOLD is set to 5%.
