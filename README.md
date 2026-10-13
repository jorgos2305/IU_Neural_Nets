# IU International University of Applied Sciences

## Course: Neural Nets and Deep Learning

Welcome to the repository of the Neural Nets and Deep Learning course. The task for this project consists in explaining how a neural network learns and what challenge is addressed with __batch normalization__.

## Conda environment

This project was implemented using a conda environment. To replicate it, first cline the repository and then run:
```bash
cd IU_Neural_Nets
conda env create -f environment.yml
conda activate tensorf
```

## Accelerating Deep Network Training via Batch Normalization

The task consists in comparing three neural networks as described in the table below

<table>
    <tr>
        <th>Experiment</th>
        <th>Architecture</th>
        <th>Normalization</th>
        <th>Optimizer</th>
        <th>Learning rate</th>
        <th>Remarks</th>
    </tr>
    <tr>
        <td>FNN-1</td>
        <td>FNN</td>
        <td>None</td>
        <td>SGD with momentum = 0.9</td>
        <td>0.001</td>
        <td>Baseline</td>
    <tr>
        <td>FNN-2</td>
        <td>FNN</td>
        <td>Batch Normalization</td>
        <td>SGD with momentum = 0.9</td>
        <td>0.001</td>
        <td></td>
    </tr>
    <tr>
        <td>FNN-3</td>
        <td>FNN</td>
        <td>Layer Normalization</td>
        <td>SGD with momentum = 0.9</td>
        <td>0.001</td>
        <td></td>
    </tr>
</table>

The baseline network needed 193 epochs to reach 85% validation accuracy. It reached the maximum validation accuracy and lowest validation loss after 246 epochs.

The batch-norm network needed 3 epochs to reach 85% validation accuracy. It reached the maximum validation accuracy and the lowest validation loss after 12 epochs (≈20 times less epochs than the baseline).

The layer-norm network needed 6 epochs to reach 85% validation accuracy. It reached the maximum validation accuracy and the lowest validation loss after 35 epochs (≈7 times less epochs than the baseline).

Additionally, the baseline network had a harder time learning since it took around 50 epoch for it to really start learning and the loss to be reduced.