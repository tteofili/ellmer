import pandas as pd

from ellmer.post_hoc.lemon_adapter import ltuple_rtuple_to_lemon_frames, make_lemon_predict_proba


def test_lemon_frames_roundtrip_keys():
    lt = {"id": 1, "name": "foo", "price": "10"}
    rt = {"id": 2, "name": "foo", "price": "11"}
    ra, rb, pairs = ltuple_rtuple_to_lemon_frames(lt, rt)
    assert len(ra) == 1 and len(rb) == 1
    assert list(pairs["a.rid"]) == [0]
    assert ra.loc[0, "name"] == "foo"


def test_predict_proba_wrapper_batch():
    def predict_fn(df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for _i in range(len(df)):
            out.append({"nomatch_score": 0.2, "match_score": 0.8})
        return pd.DataFrame(out)

    pp = make_lemon_predict_proba(predict_fn)
    ra, rb, pairs = ltuple_rtuple_to_lemon_frames({"id": 1, "x": "a"}, {"id": 2, "x": "b"})
    pairs2 = pd.DataFrame({"a.rid": [0, 0], "b.rid": [0, 0]})
    probs = pp(ra, rb, pairs2)
    assert len(probs) == 2
    assert probs[0] == 0.8
