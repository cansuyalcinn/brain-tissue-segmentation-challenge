# Brain tissue segmentation using Multi-atlas and Deep learning approaches

In this project, we performed segmentation of the three main brain tissues cerebrospinal fluid (CSF), gray matter (GM) and white matter (WM), from the
brain MRI dataset of IBSR 18.

# Implementation

We followed two main pipelines; classical approaches containing multi-atlas, bayesian model and intensity based models as well as deep learning models containing U-net, DenseUnet and ResUnet. Overall, we performed experiments to find the optimized setting for our segmentation problem. Among the mean of the validation dataset, we were able to obtain 0.80, 0.88, 0.88 dice scores, using the Bayesian model, and 0.84, 0.93 and 0.91 dice scores using the DenseUnet model for the CSF, GM and WM sequentially.
