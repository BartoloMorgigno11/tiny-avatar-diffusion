from tiny_avatar_diffusion.captions import caption_tokens, generate_caption
from tiny_avatar_diffusion.semantic_mapping import (
    NEUTRAL_TOKEN,
    OFFICIAL,
    semantic_value,
)
from tiny_avatar_diffusion.tokenizer import (
    UNK_ID,
    WhitespaceTokenizer,
    build_train_vocabulary,
    normalize_and_split,
)

SAMPLE = {"face_shape": 4, "glasses": 5, "hair_color": 6, "face_color": 8}


def test_caption_uses_frozen_order_and_atomic_tokens() -> None:
    expected = (
        "avatar face_shape_round glasses_round_glasses "
        "hair_color_dark_brown face_color_light_beige"
    )
    assert generate_caption(SAMPLE) == expected
    assert caption_tokens(SAMPLE) == expected.split()


def test_mapping_provenance_special_cases() -> None:
    assert semantic_value("glasses", 11).provenance_status == OFFICIAL
    assert semantic_value("face_shape", 1).provenance_status == NEUTRAL_TOKEN
    assert semantic_value("face_shape", 1).canonical_model_token == "face_shape_1"


def test_tokenizer_normalizes_only_case_and_whitespace() -> None:
    assert normalize_and_split("  AVATAR   face_shape_1\n") == ["avatar", "face_shape_1"]


def test_vocabulary_order_and_serialization_are_deterministic() -> None:
    caption = generate_caption(SAMPLE)
    vocabulary, _ = build_train_vocabulary([caption, caption])
    assert [vocabulary[token] for token in ("<pad>", "<bos>", "<eos>", "<unk>")] == [
        0,
        1,
        2,
        3,
    ]
    assert list(vocabulary)[4:] == sorted(vocabulary.keys() - {"<pad>", "<bos>", "<eos>", "<unk>"})
    encoded = WhitespaceTokenizer(vocabulary).encode(caption)
    assert encoded.tokens[0] == "<bos>"
    assert encoded.tokens[-1] == "<eos>"
    assert len(encoded.tokens) == len(encoded.token_ids) == len(encoded.attention_mask) == 7
    assert encoded.attention_mask == [1] * 7


def test_unknown_token_maps_to_unk_without_changing_vocabulary() -> None:
    vocabulary, _ = build_train_vocabulary(["avatar known one two three"])
    before = dict(vocabulary)
    encoded = WhitespaceTokenizer(vocabulary).encode("avatar missing one two three")
    assert UNK_ID in encoded.token_ids
    assert vocabulary == before
