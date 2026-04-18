This repository contains the code, data, and reproducible workflows for our study on Observable-Constrained Variational Framework (OCVF) for correcting prior effective models of complex materials systems. The central motivation of this work is that conventional bottom-up fitting strategies, although physically grounded, may suffer from systematic bias when finite-temperature effects, long-range interactions, and collective many-body behavior become important. As a result, a prior model that appears reasonable at the microscopic level may still fail to reproduce the correct macroscopic phase behavior.

To address this limitation, we propose OCVF as a top-down correction framework that incorporates macroscopic observable constraints into model refinement. Rather than discarding the prior model, OCVF uses experimentally or physically motivated observables to guide the correction process, thereby preserving the microscopic foundation while improving agreement with target thermodynamic behavior.

In this repository, we provide the implementation of the OCVF optimization pipeline, the datasets used in this study, and the scripts required to reproduce the main results reported in our manuscript. Using BaTiO
3
	​

 (BTO) as a representative case, the workflow demonstrates how OCVF can successfully correct deficiencies in the prior model and recover the physically correct phase-transition sequence. The repository is intended not only to support reproducibility of the present work, but also to serve as a starting point for applying observable-constrained correction strategies to other complex many-body materials systems.

We hope this resource will be useful for researchers working in computational materials science, multiscale modeling, effective Hamiltonians, phase transitions, and physics-informed machine learning.
