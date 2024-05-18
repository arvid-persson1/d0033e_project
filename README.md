These commands install the necessary packages to run the project.

```
py -m ensurepip --upgrade
pip install numpy pandas matplotlib PyQt5 scipy scikit-learn
```

There is no main file, and most of the project is designed to run from the command line.

`data/` contains all datasets in csv format, as well as results from various calculations

`optimization/` contains the results of all iterations of optimization, sorted by model.

`src/` contains all scripts. Note that some steps were carried out with temporary scripts
or tools like Excel, and are not included.
