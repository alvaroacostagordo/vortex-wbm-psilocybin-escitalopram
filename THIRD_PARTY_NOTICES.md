# Third-party notices

The MIT License in the repository root applies to repository-authored code. The files listed below retain their original attribution and remain subject to the terms attached to their source distributions.

## Brain Connectivity Toolbox

`06_network_modularity/community_louvain.m` is derived from the Brain Connectivity Toolbox implementation by Mika Rubinov. The function is used only with the Newman modularity objective at γ = 1.

- Project: [Brain Connectivity Toolbox](https://sites.google.com/site/bctnet/)
- Source repository and license: [brainlife/BCT](https://github.com/brainlife/BCT)
- Citation: Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: Uses and interpretations. *NeuroImage, 52*(3), 1059–1069. https://doi.org/10.1016/j.neuroimage.2009.10.003

The original attribution and modification history are retained in the source file.

## Gaussian-filter helper

`01_vortex_space_model/gaussfilt.m` and `02_signal_space_model/gaussfilt.m` are copies of the Gaussian-filter function by James Conder. They are included locally because the model-fitting scripts call the function directly.

- Source: [gaussfilt(t,z,sigma), MATLAB Central File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/43182-gaussfilt-t-z-sigma)
- Author: James Conder

The original author notice is retained in the source file. Redistribution and reuse of this helper are governed by the license supplied with the File Exchange submission, not by the repository MIT License.

## Schaefer parcellation-derived files

`schaefer1000to100.mat` and `schaefer100_region_labels.xlsx` encode mappings or labels derived from the Schaefer cortical parcellation. Users should cite:

Schaefer, A., Kong, R., Gordon, E. M., Laumann, T. O., Zuo, X.-N., Holmes, A. J., Eickhoff, S. B., & Yeo, B. T. T. (2018). Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. *Cerebral Cortex, 28*(9), 3095–3114. https://doi.org/10.1093/cercor/bhx179

No Human Connectome Project (HCP) structural-connectivity matrix or participant-level trial data are distributed in this repository.