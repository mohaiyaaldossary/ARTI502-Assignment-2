# ARTI 502 - Assignment 2

ARTI 502 - Assignment 2 - Group 2

FILES
- Assignment_2_Code.ipynb: complete notebook for Tasks 1-6, with saved results and figures.
- Assignment_2_Code.py: the same complete source as a Python script.
- train_improved_cnn.py: optional standalone runner for Experiment 4 only.
- experiment4_results/best_model.pt: saved model selected using validation accuracy.
- experiment4_results/history.json: measured values for all 40 epochs.
- experiment4_results/results.json: final saved-model test results.
- experiment4_results/training_curves.png: the required loss and accuracy plots.
- experiment4_results/run_manifest.json: dataset checksum and execution file hashes.

RUNNING THE COMPLETE ASSIGNMENT
Install torch, torchvision, numpy and matplotlib in your Python environment.
For GPU support, use the appropriate PyTorch installer for your hardware:
https://pytorch.org/get-started/locally/
Open the notebook in Jupyter/VS Code, or run:
    python Assignment_2_Code.py
The script downloads CIFAR-10 and runs all four experiments. It saves the new
Experiment 4 results in ./experiment4_results. Run from a folder where you
want to keep the dataset and results. Rerunning will replace results in that folder.

TO RERUN ONLY EXPERIMENT 4
Run the Experiment 4 cells in the notebook, or:
    python train_improved_cnn.py --epochs 40 --batch-size 128
The standalone runner writes the measurements and model checkpoint.
The notebook and complete script additionally generate the plots.

MEASURED RESULT
40 epochs completed on an NVIDIA RTX 3060 Laptop GPU.
Validation-selected model: epoch 39.
Test accuracy: 88.62% (8862 / 10000 correct).
Test loss: 0.3418329833984375.
Final epoch 40 test accuracy: 88.55%.
Original Experiment 3 final test accuracy: 77.31%.
Improvement: 11.31 percentage points.
The new run uses 45,000 training images and 5,000 validation images;
the original runs use all 50,000 training images. No pretrained weights are used.
Test curves are logged as required by the PDF; checkpoint selection uses validation.
Reruns may differ across environments even with a fixed seed.

GROUP MEMBERS
Mohammed Jaarran - 2230000129
Mohaiya Aldossary - 2230004579
Sameer Almadani - 2230001818
Omar Bodai - 2240040094
Hussain Alabdurabalnabi - 2230004638

SUBMISSION
Group repository: https://github.com/mohaiyaaldossary/ARTI502-Assignment-2
The repository contains the new code, report, trained model and measured results.
Submit Assignment_2_Report.pdf with the source-code repository link.
