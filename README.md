\# CObL: Concurrent Object Layers - Inference Implementation



This repository contains the inference code for the CObL project. It implements the "Ordinal Layering" and "Recursive Compositing" algorithms described in the paper.



\## 📂 Project Structure

\- `CObl\_model/`: Contains the decomposition logic (Amodal Segmentation \& Inpainting).

\- `cobl.py`: \*\*\[CORE]\*\* Implements the Recursive Compositing equation (Eq. 1).

\- `main\_inference.py`: Runs the pipeline using MapReduce for large datasets.

\- `results/`: Output directory.



\## 🚀 How to Run

1\. Install dependencies:

&nbsp;  ```bash

&nbsp;  pip install -r requirements.txt

