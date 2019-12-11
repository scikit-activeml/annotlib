from annotlib.base import BaseAnnot
from annotlib.cluster_based import ClusterBasedAnnot
from annotlib.classifier_based import ClassifierBasedAnnot
from annotlib.dynamic import DynamicAnnot
from annotlib.multi_types import MultiAnnotTypes
from annotlib.difficulty_based import DifficultyBasedAnnot
from annotlib.standard import StandardAnnot

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")