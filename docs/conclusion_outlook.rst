Conclusion and Outlook
======================

In this guide, we presented our software package *annotlib* modelling annotators in an active learning cycle.
Therefor, we illustrated the active learning cycle in combination with multiple annotators
and showed the structure of our package, which consists of seven annotator types implemented as Python classes.
To reflect real annotators, a created instance of an annotator type offers a collection of functions, e.g,
providing class labels for given samples.
We explained the functions of different annotator types by means of *Jupyter Notebooks*.
Each of these notebooks comprises a description of the corresponding annotator type and code examples illustrating the
employment of annotators in combination with exemplary data sets.
Moreover, we analysed generated annotators regarding their labelling performances and
distributions of provided class labels as well as confidence scores with help of functions implemented
by our *annotlib* package.
Four of the seven annotator types represent simulation techniques reflecting real-world scenarios.
As a result, adversarial and non-adversarial annotators with different knowledge areas, considered decision criteria,
and kinds of decision making can be simulated.
To simulate dynamic labelling performances, we introduced learning rates which modify the labelling accuracy depending
on the number of samples labelled by an annotator.
We proposed also a procedure to estimate the difficulty to label a sample correctly, so that annotators whose labelling
performances depend on these difficulties can be simulated.
The simulation techniques aim at overcoming the lack of publicly available data sets labelled by multiple
annotators.
Furthermore, a library of simulation techniques enables a standardised evaluation of active learning strategies coping
with multiple annotators.

In future work, the Python package *annotlib* is to be applied in projects where active learning strategies take
error-prone multiple annotators into account.
Hence, we will publish *annotlib* to the Github platform as well as the package management system of Python, so that
users are able to integrate *annotlib* into their Python environment.
An already existing HTML version of this guide will be published as website to ease the usability of *annotlib*.
Based on the user's experiences in projects, *annotlib* is to be continuously maintained and improved.
The maintenance will guaranty a reliable and stable version of *annotlib*, whereas improvements may lead to
extensions of our package by adding new annotator types as well as simulation techniques.