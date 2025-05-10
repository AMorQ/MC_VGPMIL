# MC_VGPMIL

MC_VGPMIL stands for **Multiclass Variational Gaussian Processes for Multiple Instance Learning**. This repository implements a framework for multiclass classification using Variational Gaussian Processes within the context of Multiple Instance Learning (MIL). It is written entirely in Python and aims to provide a robust and flexible solution for research and practical applications in MIL.

*Attention: this is a toy model implementation*

## Table of Contents

-   Introduction
-   Features
-   Setup and Installation
-   Usage
-   Examples

## Introduction

**MC_VGPMIL** leverages Variational Gaussian Processes (GPs) to handle the complexities of Multiple Instance Learning (MIL) tasks, where labels are assigned to sets (or "bags") of instances rather than individual data points. This framework extends Gaussian Processes to solve multiclass problems, making it highly versatile for a variety of real-world applications, such as:

* Medical Imaging
* Text Classification
* Object Detection in Images

The repository provides tools for training and evaluating models, as well as examples for real-world datasets.

## Features

* **Multiclass Variational Gaussian Processes:**
    * Implemented for MIL tasks.
    * Flexible and scalable for large datasets.
* **Multiple Instance Learning Framework:**
    * Handles bags of instances effectively.
    * Supports both instance-level and bag-level inference.
* **Python Implementation:**
    * Fully written in Python for ease of use and integration.
* **Customizable Models:**
    * Easily adapt the framework to specific MIL tasks.

## Setup and Installation

### Prerequisites

* Python 3.8 or newer.
* Package manager (`pip` or `conda`).

### Installation Steps

1.  Clone the repository:

    ```bash
    git clone [https://github.com/AMorQ/MC_VGPMIL.git](https://github.com/AMorQ/MC_VGPMIL.git)
    cd MC_VGPMIL
    ```

2.  Set up a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Prepare Your Dataset:

Organize your data in a bag-label format suitable for MIL tasks. Follow the provided examples to preprocess your dataset.

### Train the Model:

Use the training scripts to fit the **MC_VGPMIL** model to your data. Example command:

```bash
python train.py --data_path <path_to_data> --output_dir <output_directory>
```

### Evaluate the Model:

Assess the model's performance on validation or test datasets. Example command:

```bash
python evaluate.py --model_path <path_to_model> --data_path <path_to_data>
```

### Customize the model:

Edit the `config.json` file
