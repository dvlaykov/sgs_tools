# Python tools for SGS analysis

## Install

  Use typical `pip` installation (preferably within a virtual environment to keep dependencies clean) e.g.

  ``` pip install git+https://github.com/dvlaykov/sgs_tools.git```

**NB** The package is in active development. No backwards compatibility is guarranteed at this time.

## Documentation
The docs are generated via [sphinx](https://www.sphinx-doc.org/en/master/) and can be generated locally by
1. Downloading the [repository](https://github.com/dvlaykov/sgs_tools)
2. Running `make doc` from the top level in the repository will generate html version of the documentation. (you will need the `sphinx`, `sphinx-mdinclude`, and `pydata-sphinx-theme` python packages in your environment)
3. The resultind docs can be accessed from `<repo_directory>/doc/_build/html/index.html`
