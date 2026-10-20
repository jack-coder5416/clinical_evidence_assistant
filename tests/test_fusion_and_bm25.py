from src.retriever.bm25 import BM25Index, tokenize, query_terms
from src.retriever.fusion import reciprocal_rank_fusion


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion({"dense": ["a", "b", "c"], "sparse": ["d", "b", "e"]}, k=60)
    order = [d for d, _ in fused]
    assert order[0] == "b"                     # 2nd in both lists beats 1st in only one
    assert set(order) == {"a", "b", "c", "d", "e"}


def test_rrf_exact_score():
    fused = dict(reciprocal_rank_fusion({"x": ["a"], "y": ["a"]}, k=60))
    assert fused["a"] == 2 / 61


def test_rrf_weights():
    fused = reciprocal_rank_fusion({"dense": ["a"], "sparse": ["b"]}, weights={"dense": 1, "sparse": 3})
    assert fused[0][0] == "b"


def test_tokenizer_collapses_spelling_and_plurals():
    assert tokenize("randomised trials") == tokenize("randomized trial")
    assert tokenize("randomisation") == tokenize("randomization") == tokenize("randomized")
    assert tokenize("oedema") == tokenize("edema")
    assert tokenize("paediatric") == tokenize("pediatric")
    assert tokenize("behaviour") == tokenize("behavior")
    assert "the" not in tokenize("the effect of the drug")


def test_bm25_exact_acronym_match():
    idx = BM25Index(["a", "b", "c"], [
        "blinding reduces bias", "a CTIMP is an investigational medicinal product trial", "sample size"])
    assert idx.search("CTIMP", 3)[0][0] == "b"


def test_bm25_empty_and_stopword_only_query():
    idx = BM25Index(["a"], ["some text here"])
    assert idx.search("", 5) == [] and idx.search("the of and", 5) == []


def test_query_terms_for_highlighting():
    assert query_terms("What is the CTIMP?") == ["ctimp"]
