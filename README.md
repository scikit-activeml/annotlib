# annotlib: Simulation of Annotators

authors: Marek Herde and Adrian Calma

![](https://github.com/acalma/annotlib/workflows/annotlib/badge.svg)

# Introduction

*annotlib* is a Python package for simulating annotators in an active learning setting.
Solving classification problems by using supervised machine learning models requires samples assigned to class labels.
However, labeling these samples causes costs (e.g. workload, time, etc.), so that active learning strategies aim at reducing these costs by selecting samples, which are the most useful for training a classifier.

In real-world scenarios, human annotators are often responsible for providing the class labels.
Unfortunately, there is no guaranty for the omniscience of such annotators.
Hence, annotators are prone to error, respectively uncertain, so that the class labels assigned to samples may be false.
The labeling performance of an annotator is affected by many factors, e.g. expertise, experience, concentration, level of fatigue and so on.
Moreover, the difficulty of a sample influences the outcome of a labeling process.

To evaluate an active learning strategy in the setting of uncertain annotators, class labels of these uncertain
annotators are required to be available, but there is a lack of publicly accessible real-world data sets labeled by error-prone annotators.
As a result, recently published active learning strategies are evaluated on simulated annotators where the used simulation techniques are diverse.
Our developed *annotlib* Python package represents a of these techniques and implements additional methods, which simulate realistic characteristics of uncertain annotators.
This way, we establish a library simplifying and standardising the evaluation of active learning strategies coping with uncertain annotators.

For more information go to the [documentation](https://annotlib.readthedocs.io).
