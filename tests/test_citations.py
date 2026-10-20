from src.citations import analyze, decorate


def test_valid_and_invalid_citations():
    rep = analyze("Consent is required [2]. Blinding helps [1][3]. Made-up claim [9].", n_sources=3)
    assert rep.cited == [2, 1, 3] and rep.invalid == [9]


def test_alternative_formats_are_understood():
    rep = analyze("A [1, 2]. B [Source 3]. C [1-3]. D [Sources 1 and 2].", n_sources=3)
    assert set(rep.cited) == {1, 2, 3} and not rep.invalid


def test_uncited_sentence_flagged_but_short_and_headers_not():
    text = ("Key points:\n- The guidance recommends proportionate monitoring of all trials in practice.\n"
            "- Participants may withdraw at any time without penalty or loss of benefit [2].")
    rep = analyze(text, n_sources=2)
    assert len(rep.uncited_sentences) == 1 and "proportionate monitoring" in rep.uncited_sentences[0]


def test_decorate_marks_invalid_differently():
    out = decorate("Yes [1]. No [7].", n_sources=2)
    assert ":blue-badge[1]" in out and ":red-badge[?7]" in out


def test_no_false_citation_from_ordinary_brackets():
    rep = analyze("Values were in the range [0.5, 2.0] and the arm [A] was used in trial.", n_sources=3)
    assert rep.cited == [] and rep.invalid == []


def test_tidy_puts_citations_after_sentence_punctuation():
    from src.citations import tidy
    assert tidy("Consent is voluntary. [1]. Next point.") == "Consent is voluntary. [1] Next point."
    assert tidy("Already fine [1]. Untouched.") == "Already fine. [1] Untouched."
    assert tidy("A [1][2]. B [3, 4].") == "A. [1][2] B. [3, 4]"
    assert tidy("Canonical already. [1] Next.") == "Canonical already. [1] Next."
    assert tidy("The value was 0.5 units.") == "The value was 0.5 units."          # untouched
    assert tidy(tidy("x [1].")) == tidy("x [1].")                                   # idempotent


def test_citation_stays_attached_to_its_own_sentence():
    rep = analyze("Participants may withdraw at any time without penalty or reason. [1]. "
                  "Consent must be voluntary and informed at all stages of the trial. [2] "
                  "This second claim has no citation but is long enough to be flagged here.", n_sources=2)
    assert rep.cited == [1, 2]
    assert len(rep.uncited_sentences) == 1 and rep.uncited_sentences[0].startswith("This second claim")


def test_decorate_renders_badge_after_punctuation():
    out = decorate("Participants may withdraw [1]. Then more. [2]", n_sources=2)
    assert out == "Participants may withdraw. :blue-badge[1] Then more. :blue-badge[2]"
