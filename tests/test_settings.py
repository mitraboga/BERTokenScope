from ber_tokenscope.settings import Settings


def test_service_settings_have_deployment_defaults():
    settings = Settings()

    assert settings.service.api_port == 8788
    assert settings.service.dashboard_port == 8501
    assert settings.service.worker_concurrency == 1
