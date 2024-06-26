installation
============

Installation
------------

The AIPyS platform is engineered to streamline the integration of AI with photoswitchable genetic CRISPR screen technology. Presently, AIPyS supports installation on Windows systems and requires Python 3.8. To ensure optimal functionality, particularly for components like PyTorch and Cellpose, matching the CUDA and cuDNN versions to your system’s configuration is crucial.

Prerequisites
-------------

Before proceeding with the installation, ensure your system meets the following requirements:

- Operating System: Windows
- Python Version: 3.8

Additionally, for functionalities leveraging PyTorch and Cellpose, verify you have the appropriate versions of CUDA and cuDNN installed. These components are essential for computation offloading to GPUs, significantly enhancing performance for machine learning tasks.

Installation Options
--------------------

AIPyS offers two primary methods for installation: via an ``environment.yml`` file using Conda and directly through pip.

Option 1: Using ``environment.yml`` with Conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For users preferring Conda, AIPyS provides an ``environment.yml`` file that simplifies the setup process, managing both Python version constraints and necessary library dependencies, including CUDA and cuDNN configurations.

1. Download the ``environment.yml`` file from the AIPyS repository.
2. Open a Windows command prompt or PowerShell window.
3. Navigate to the directory containing the downloaded ``environment.yml`` file.
4. Execute the following command:

.. code-block:: bash

   conda env create -f environment.yml

5. Activate the newly created Conda environment:

.. code-block:: bash

   conda activate aipys_env

This environment will include all the necessary dependencies pre-configured, adhering to the CUDA and cuDNN versions required for PyTorch and Cellpose.

Option 2: Installation with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, AIPyS can be installed using pip, which is Python’s package manager. This method is straightforward but requires manual management of compatible CUDA and cuDNN versions.

.. code-block:: bash

   pip install --upgrade AIPyS

Before executing the install command, ensure your Python environment is configured with Python 3.8. As AIPyS is currently optimized for Windows, users on other platforms should carefully consider compatibility, especially regarding CUDA and cuDNN dependencies.

Verifying Installation
----------------------

After completing the installation using either method, you can verify the setup by checking the installed AIPyS version:

.. code-block:: bash

   aipys --version

Alternatively, attempt running a simple command or script to confirm operational functionality.

Troubleshooting
---------------

- **CUDA/cuDNN Compatibility:** If you encounter issues related to PyTorch or Cellpose, verify that you have installed the correct versions of CUDA and cuDNN that match the requirements of these libraries.
- **Python Version:** Ensure that the Python environment active during installation is version 3.8, as discrepancies in Python versions can lead to compatibility problems.
- **GPU Recognition Issues:** If your GPU is not recognized and you are using an NVIDIA Quadro P4000, ensure that you have the appropriate drivers installed. Additionally, install PyTorch with a CUDA version that matches your setup by executing the following command:

  .. code-block:: bash

     pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

  This command installs the versions of PyTorch, torchvision, and torchaudio that are compatible with CUDA 11.3, specifically optimized for the NVIDIA Quadro P4000 GPU.

Evaluate PyTorch-GPU
--------------------

  .. code-block:: python

      import torch 
      print(torch.version.cuda)  



For further assistance, refer to the FAQs section or reach out to the AIPyS support team.
