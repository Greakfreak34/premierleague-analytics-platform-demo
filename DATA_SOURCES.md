# Data and third-party notices

The default demo uses only `tools/build_demo.py`, NumPy's seeded generator, and
fictional club names. No football dataset or website content is distributed.
No Premier League, FPL, Sports Mole, Football-Data, OpenFootball, Understat or
Bluesky responses are included or fetched by setup.

Optional sentiment scoring references
[CardiffNLP Twitter RoBERTa sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest),
revision `3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7`, whose model card declares
CC-BY-4.0. Weights are not shipped. The scoring implementation follows the card's
mention/URL preprocessing. Loading the model is optional and subject to its license.

Python dependencies retain their respective licenses. This repository does not
relicense those packages or grant rights to any external football data. No broad
software reuse license has been selected for this public code yet; public
visibility alone is not an unrestricted reuse grant.
