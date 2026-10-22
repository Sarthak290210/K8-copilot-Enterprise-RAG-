from scripts import seed_db


def test_corpus_selection_is_reproducible():
    signal_a, noise_a = seed_db._select_corpus(10)
    signal_b, noise_b = seed_db._select_corpus(10)
    assert noise_a == noise_b  # seeded: every run and every eval uses the same noise sample
    assert len(noise_a) == 10
    assert signal_a == seed_db._collect_files("true_data")  # all signal docs, always


def test_all_and_oversized_samples_take_everything():
    _, everything = seed_db._select_corpus("all")
    _, oversized = seed_db._select_corpus(len(everything) + 5)
    assert everything == oversized == seed_db._collect_files("noisy_data")


def test_collect_files_skips_unsupported():
    files = seed_db._collect_files("true_data")
    assert files and all(f.suffix.lower() in {".pdf", ".docx", ".html", ".txt"} for f in files)
    assert all(f.name != ".gitkeep" for f in files)
