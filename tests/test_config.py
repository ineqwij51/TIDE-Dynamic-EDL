from tide.config import load_config


def test_public_configuration_loads() -> None:
    config = load_config("configs/tide.yaml")
    assert config["model"]["hidden_dim"] == 48
    assert config["data"]["window_seconds"] == 20
    assert config["training"]["epochs"] == 20
    assert config["loss"]["lambda_bridge"] == 0.10

