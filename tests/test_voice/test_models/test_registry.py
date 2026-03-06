"""Tests for ModelRegistry — registration, lookup, defaults."""

from alchemy.voice.models.registry import ModelRegistry
from alchemy.voice.models.schemas import ModelCapability, ModelCard, ModelLocation, SpeedTier


def _make_card(name: str, caps: list[ModelCapability], **kwargs) -> ModelCard:
    return ModelCard(
        name=name,
        capabilities=caps,
        speed_tier=kwargs.get("speed_tier", SpeedTier.FAST),
        location=kwargs.get("location", ModelLocation.GPU_LOCAL),
        **{k: v for k, v in kwargs.items() if k not in ("speed_tier", "location")},
    )


class TestRegister:
    def test_register_and_get(self):
        reg = ModelRegistry()
        card = _make_card("test:14b", [ModelCapability.CONVERSATION])
        reg.register(card)
        assert reg.get("test:14b") is card

    def test_get_missing_returns_none(self):
        reg = ModelRegistry()
        assert reg.get("nope") is None

    def test_unregister(self):
        reg = ModelRegistry()
        reg.register(_make_card("m1", [ModelCapability.CONVERSATION]))
        reg.unregister("m1")
        assert reg.get("m1") is None

    def test_unregister_clears_defaults(self):
        reg = ModelRegistry()
        reg.register(
            _make_card(
                "m1",
                [ModelCapability.CONVERSATION],
                is_default_for=[ModelCapability.CONVERSATION],
            )
        )
        reg.unregister("m1")
        assert reg.get_default(ModelCapability.CONVERSATION) is None


class TestDefaults:
    def test_default_set_on_register(self):
        reg = ModelRegistry()
        card = _make_card(
            "conv",
            [ModelCapability.CONVERSATION],
            is_default_for=[ModelCapability.CONVERSATION],
        )
        reg.register(card)
        assert reg.get_default(ModelCapability.CONVERSATION) is card

    def test_default_not_set_without_flag(self):
        reg = ModelRegistry()
        reg.register(_make_card("conv", [ModelCapability.CONVERSATION]))
        assert reg.get_default(ModelCapability.CONVERSATION) is None

    def test_multiple_defaults(self):
        reg = ModelRegistry()
        reg.register(
            _make_card(
                "conv",
                [ModelCapability.CONVERSATION, ModelCapability.CLASSIFICATION],
                is_default_for=[ModelCapability.CONVERSATION, ModelCapability.CLASSIFICATION],
            )
        )
        assert reg.get_default(ModelCapability.CONVERSATION) is not None
        assert reg.get_default(ModelCapability.CLASSIFICATION) is not None


class TestLookup:
    def test_find_by_capability(self):
        reg = ModelRegistry()
        reg.register(_make_card("a", [ModelCapability.CONVERSATION]))
        reg.register(_make_card("b", [ModelCapability.VISION]))
        reg.register(_make_card("c", [ModelCapability.CONVERSATION, ModelCapability.VISION]))

        conv_models = reg.find_by_capability(ModelCapability.CONVERSATION)
        assert len(conv_models) == 2
        names = {m.name for m in conv_models}
        assert names == {"a", "c"}

    def test_all_models(self):
        reg = ModelRegistry()
        reg.register(_make_card("a", [ModelCapability.CONVERSATION]))
        reg.register(_make_card("b", [ModelCapability.VISION]))
        assert len(reg.all_models()) == 2

    def test_endpoint_stored(self):
        reg = ModelRegistry()
        card = _make_card(
            "dual",
            [ModelCapability.CONVERSATION],
            endpoint="http://wsl-host:11434",
        )
        reg.register(card)
        assert reg.get("dual").endpoint == "http://wsl-host:11434"
