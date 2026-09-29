from bidding_dt.data.vocab import (CALL_PAD, CALL_PASS, CALL_X, CALL_XX,
                                   FIRST_BID, VOCAB_SIZE, bid_id, call_to_id,
                                   id_to_call)


def test_vocab_size():
    assert VOCAB_SIZE == 39
    ids = {call_to_id(c) for c in
           ["P", "X", "XX"] + [f"{l}{d}" for l in range(1, 8) for d in "CDHSN"]}
    assert len(ids) == 38
    assert min(ids) == CALL_PASS and CALL_PAD not in ids


def test_roundtrip():
    for i in range(VOCAB_SIZE):
        assert call_to_id(id_to_call(i)) == i


def test_bid_ordering():
    # ids strictly follow bridge ranking: level-major, C<D<H<S<N
    seq = [bid_id(l, d) for l in range(1, 8) for d in "CDHSN"]
    assert seq == sorted(seq)
    assert call_to_id("1N") > call_to_id("1S")
    assert call_to_id("2C") > call_to_id("1N")
    assert call_to_id("7N") == VOCAB_SIZE - 1
    assert FIRST_BID == call_to_id("1C")
    assert CALL_XX == FIRST_BID - 1
