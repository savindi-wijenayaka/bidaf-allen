from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz")

print()
print()
print("================= MODEL OUTPUTS =================")
print()
json_result = predictor.predict(
  passage="The Matrix is a 1999 science fiction action film written and directed by The Wachowskis, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano.",
  question="Who stars in The Matrix?"
)
print()
print("ANSWER : "+json_result["best_span_str"])
print()
print("================ END OF OUTPUTS =================")
print()