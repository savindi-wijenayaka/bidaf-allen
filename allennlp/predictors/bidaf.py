from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('machine-comprehension')
class BidafPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.BidirectionalAttentionFlow` model.
    """

    def predict(self, question: str = None, passage: str = None) -> JsonDict:
        """
        Make a machine comprehension prediction on the supplied input.
        See https://rajpurkar.github.io/SQuAD-explorer/ for more information about the machine comprehension task.

        Parameters
        ----------
        question : ``str``
            A question about the content in the supplied paragraph.  The question must be answerable by a
            span in the paragraph.
        passage : ``str``
            A paragraph of information relevant to the question.

        Returns
        -------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        return self.predict_json({"passage" : passage, "question" : question})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        question_text=None
        passage_text=None
        if json_dict["question"] != None:
            question_text = json_dict["question"]
        if json_dict["passage"] != None: 
            passage_text = json_dict["passage"]

        if question_text and passage_text:
            # print("ok")
            return self._dataset_reader.text_to_instance(question_text=question_text, passage_text=passage_text)
        elif question_text and passage_text is None:
            # print("ok1")
            tokenized_question = self._dataset_reader.text_to_instance_one_argument(passage_text=question_text)
            # print(tokenized_question)
            return tokenized_question
        elif passage_text and question_text is None:
            print(passage_text)
            return self._dataset_reader.text_to_instance_one_argument(passage_text=passage_text)
        else:
            raise ValueError('Both question and passage is missing')
