# magnitude_dncnn

Code used for denoising T2star-weighted susceptibility imaging.

---

## Getting started

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)

### Installation

1. **Clone the repository**

```bash
   git clone https://github.com/Cedars-Sinai-Neuroimaging/magnitude_dncnn.git
   cd magnitude_dncnn
```

2. **Create the conda environment**
   ```bash
   conda env create -f environment.yml
   ```
 
3. **Activate the environment**
   ```bash
   conda activate magnitude_dncnn
   ```
 
4. **Verify the installation** *(optional)*
   ```bash
   conda env list
   ```

---

### Denoising a Nitfi Image
 
```python
   python denoise.py input.nii.gz output.nii.gz models/dncnn.h5
```
Assumes sagittal axis is aligned along the Z-axis for a given image of dimensions [X, Y, Z]. To specify a different dimension for the sagittal axis, add flag `--sagittal_axis [1,2,3]`.

---

### Project Structure
 
```
magnitude_dncnn/
├── README.md
├── environment.yml     # Conda environment definition
├── train_network.py    # script used for training the model
├── models.py           # definition for DnCNN model
├── utils.py            # functions for the program
├── denoise.py          # program to denoise a given nifti image
├── models.py           # definition for DnCNN model
├── Quantitative_measures_w_wo_Denoise.m    # quantitative measures
└── Quantmeasures_raincloud_plots.ipynb     # plots

```
 
---

## License
 
Distributed under the MIT License. See `LICENSE` for more information.
