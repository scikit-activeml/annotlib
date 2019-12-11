# sim_annotator_lib: Simulation of Annotators

authors: Marek Herde and Adrian Calma

# Introduction

*sim_annotator_lib* is a library simulating annotators in an active learning setting.
Solving classification problems by using supervised machine learning models requires samples assigned to class labels.
However, labelling these samples causes cost (e.g. workload, time, etc.), so that active learning strategies aim at
reducing these cost by selecting samples, which are the most useful for training a classifier.

In real-world scenarios, human annotators are often responsible for providing the class labels.
Unfortunately, there is no guaranty for the omniscience of such annotators.
Hence, annotator are prone to error respectively uncertain, so that the class labels assigned to samples may be
false.
The labelling performance of an annotator is affected by many factors, e.g. expertise, experience, concentration,
level of fatigue and so on.
Moreover, the difficulty of a sample influences the outcome of a labelling process.

To evaluate an active learning strategy in the setting of uncertain annotators, class labels of these uncertain
annotators are required to be available, but there is a lack of data sets offering this option.
As a result, recently published active learning strategies are evaluated on simulated annotators where the used
simulation techniques are diverse.
Our developed *sim_annotator_lib* represents a Python library of these techniques and implements additional methods,
which simulate realistic characteristics of uncertain annotators.
This way, we establish a library simplifying and standardising the evaluation of active learning strategies coping
with uncertain annotators.

For more information go to the [documentation](./docs/_build/html/index.html).