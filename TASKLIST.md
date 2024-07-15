
## For developers:

### Install
  1. clone the [repository](https://github.com/dvlaykov/sgs_tools)
  2. make an editable install via
    ```pip install --editable <location-of-repository>```

### Contribute
  * You can add any issues, feature requests and pull requests directly at the
    [GitHub repository](https://github.com/dvlaykov/sgs_tools) -- all are very welcome.
  * Testing, formatting and static types are handled through `tox` and for the whole repo you can run `make test` to check formatting type hints and run the tests and their coverage.

    * Tests rely on the `pytest` package and should go to `test/test_*.py`
    * Formatting is taken care of by `ruff`. Run `make format` to auto-format.
    * Type hints are handled by `mypy`. Run `make mypy` to check for any issues


### TASKLIST:
  * **Documentation** -- comlete coverage
  * **Unit testing** -- increase coverage
  * Add scripts, CI programmes and notebook examples

  * IO
    * ~~ UM ~~
    * MONC
    * PTerodaCTILES ??
    * ??

  * Geometry
    * Integrate staggered and non-staggered grid interfaces
    * Implement gradients as methods on grids

  * Physics
    * Integrate with [MONC_utils](https://github.com/ParaConUK/monc_utils)
    * Integrate  SGS Fluxes analysis from P. Burns

  * Analysis tools
    * Add spectra calculation
    * Add Histograms/PDFs and violin calculators
    * Add tensor invariant calculators
    * Add quadrant analysis support

  * SGS
    * Add a mixed and nonlinear model extensions
    * Generalise dynamic coefficient calculations

      * test `xr.dot` behaves as desired to contract dimensions
      * Optimize simultaneous calculation at multiple scales or contraction dims




