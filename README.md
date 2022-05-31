LLAMA – Low-Level Abstraction of Memory Access
==============================================

[![ReadTheDocs](https://img.shields.io/badge/Docs-Read%20the%20Docs-blue.svg)](https://llama-doc.readthedocs.io)
[![Doxygen](https://img.shields.io/badge/API-Doxygen-blue.svg)](https://alpaka-group.github.io/llama)
[![Language](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)](https://isocpp.org/)
[![Paper](https://img.shields.io/badge/Paper-Wiley%20Online%20Library-blue.svg)](https://doi.org/10.1002/spe.3077)
[![Preprint](https://img.shields.io/badge/Preprint-arXiv-blue.svg)](https://arxiv.org/abs/2106.04284)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5901242.svg)](https://doi.org/10.5281/zenodo.5901242)
[![codecov](https://codecov.io/gh/alpaka-group/llama/branch/develop/graph/badge.svg?token=B94D9G96FA)](https://codecov.io/gh/alpaka-group/llama)

![LLAMA](docs/images/logo_400x169.png)

LLAMA is a cross-platform C++17 template header-only library for the abstraction of memory
access patterns. It distinguishes between the view of the algorithm on
the memory and the real layout in the background. This enables performance
portability for multicore, manycore and gpu applications with the very same code.

In contrast to many other solutions LLAMA can define nested data structures of
arbitrary depths. It is not limited to struct of array and array of struct
data layouts but also capable to explicitly define memory layouts with padding, blocking,
striding or any other run time or compile time access pattern.

To achieve this goal LLAMA is split into mostly independent, orthogonal parts
completely written in modern C++17 to run on as many architectures and with as
many compilers as possible while still supporting extensions needed e.g. to run
on GPU or other many core hardware.

Documentation
-------------

The user documentation can be found here:
https://llama-doc.rtfd.io.
It includes:

* Installation instructions
* Motivation and goals
* Overview of concepts and ideas
* Descriptions of LLAMA's constructs

Doxygen generated API documentation is located here:
https://alpaka-group.github.io/llama/.

We submitted a scientific preprint on LLAMA to arXiv here:
https://arxiv.org/abs/2106.04284.

We gave a talk on LLAMA at CERN's Compute Accelerator Forum on 2021-05-12.
The video recording (starting at 40:00) and slides are available here:
https://indico.cern.ch/event/975010/.

Contributing
------------

We greatly welcome contributions to LLAMA.
Rules for contributions can be found in [CONTRIBUTING.md](CONTRIBUTING.md).

Attribution
-----------

If you use LLAMA for scientific work, please consider citing this project.
We upload all releases to [zenodo](https://zenodo.org/record/4911494), where you can export a citation in your preferred format.
We provide a DOI for each release of LLAMA.

License
-------

LLAMA is licensed under the [LGPL3+](LICENSE).
