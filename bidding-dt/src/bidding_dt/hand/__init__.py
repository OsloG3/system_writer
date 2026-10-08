"""Hand inference: VQ encoder-decoder that guesses one player's 13 cards.

- model.py:  encoder over the public auction -> k VQ codes; autoregressive
             decoder that samples a whole hand from the codes alone
- data.py:   auction stores (real cache or generated) -> prefix views
- gen.py:    generate training auctions by letting the bidding models play
- train.py:  training entrypoint (generate, train, eval, checkpoint)
- sample.py: inference CLI, incl. masking cards the caller knows are gone
"""
