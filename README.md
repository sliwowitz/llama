LLAMA – Low-Level Abstraction of Memory Access
==============================================

[![ReadTheDocs](https://img.shields.io/badge/Docs-Read%20the%20Docs-blue.svg)](https://llama-doc.readthedocs.io)
[![Doxygen](https://img.shields.io/badge/API-Doxygen-blue.svg)](https://alpaka-group.github.io/llama)
[![Language](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)](https://isocpp.org/)
[![Paper](https://img.shields.io/badge/Paper-Wiley%20Online%20Library-blue.svg)](https://doi.org/10.1002/spe.3077)
[![Preprint](https://img.shields.io/badge/Preprint-arXiv-blue.svg)](https://arxiv.org/abs/2106.04284)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5901241.svg)](https://doi.org/10.5281/zenodo.5901241)
[![codecov](https://codecov.io/gh/alpaka-group/llama/branch/develop/graph/badge.svg?token=B94D9G96FA)](https://codecov.io/gh/alpaka-group/llama)

![LLAMA](docs/images/logo_400x169.png)

LLAMA is a cross-platform C\++17/C\++20 header-only template library for the abstraction of data layout and memory access.
It separtes the view of the algorithm on the memory and the real data layout in the background.
This allows for performance portability in applications running on heterogeneous hardware with the very same code.

Documentation
-------------

Our extensive user documentation is available on [Read the Docs](https://llama-doc.rtfd.io).
It includes:

* Installation instructions
* Motivation and goals
* Overview of concepts and ideas
* Descriptions of LLAMA's constructs

An API documentation is generated by [Doxygen](https://alpaka-group.github.io/llama/) from the C++ source.
Please read the documentation on Read the Docs first!

Supported compilers
-------------------

LLAMA tries to stay close to recent developments in C++ and so requires fairly up-to-date compilers.
The following compilers are supported by LLAMA and tested as part of our CI:


| Linux                                                                                         | Windows                                             | MacOS                            |
|-----------------------------------------------------------------------------------------------|-----------------------------------------------------|----------------------------------|
| g++ 10 - 13 </br> clang++ 12 - 17 </br> icpx (latest) </br> nvc++ 23.5 </br> nvcc 11.6 - 12.3 | Visual Studio 2022 </br> (latest on GitHub actions) | clang++ </br> (latest from brew) |


Single header
-------------

We create a single-header version of LLAMA on each commit,
which you can find on the [single-header branch](https://github.com/alpaka-group/llama/tree/single-header).

This also useful, if you would like to play with LLAMA on Compiler explorer:
```c++
#include <https://raw.githubusercontent.com/alpaka-group/llama/single-header/llama.hpp>
```

Contributing
------------

We greatly welcome contributions to LLAMA.
Rules for contributions can be found in [CONTRIBUTING.md](CONTRIBUTING.md).

Scientific publications
-----------------------

We published an [article](https://doi.org/10.1002/spe.3077) on LLAMA in the journal of Software: Practice and Experience.
We gave a talk on LLAMA at CERN's Compute Accelerator Forum on 2021-05-12.
The video recording (starting at 40:00) and slides are available here on [CERN's Indico](https://indico.cern.ch/event/975010/).
Mind that some of the presented LLAMA APIs have been renamed or redesigned in the meantime.

We presented recently added features to LLAMA at the ACAT22 workshop as a [poster](https://indico.cern.ch/event/1106990/contributions/5096939/)
and a contribution to the [proceedings](https://arxiv.org/abs/2302.08251).
Additionally, we gave a [talk](https://indico.cern.ch/event/1106990/contributions/4991259/) at ACAT22 on LLAMA's instrumentation capabilities during a case study on [AdePT](https://github.com/apt-sim/AdePT),
again, with a contribution to the [proceedings](https://arxiv.org/abs/2302.08252).

Attribution
-----------

If you use LLAMA for scientific work, please consider citing this project.
We upload all releases to [Zenodo](https://zenodo.org/record/4911494),
where you can export a citation in your preferred format.
We provide a DOI for each release of LLAMA.
Additionally, consider citing the [LLAMA paper](https://doi.org/10.1002/spe.3077).

License
-------

LLAMA is licensed under the [MPL-2.0](LICENSE).
