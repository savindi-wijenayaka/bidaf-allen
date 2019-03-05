"""
A :class:`~allennlp.data.dataset_readers.dataset_reader.DatasetReader`
reads a file and converts it to a collection of
:class:`~allennlp.data.instance.Instance` s.
The various subclasses know how to read specific filetypes
and produce datasets in the formats required by specific models.
"""

# pylint: disable=line-too-long
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from allennlp.data.dataset_readers.reading_comprehension import SquadReader

# from allennlp.data.dataset_readers.semantic_parsing.quarel import QuarelDatasetReader
