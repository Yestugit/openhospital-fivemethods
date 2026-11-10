from mematlas.methods.awm import (
    AWMEnvironment,
    BrowserState,
    configure_environment_factory,
    run,
    run_agent,
)


def test_public_package_exports_environment_contract_and_agent_helper():
    assert AWMEnvironment is not None
    assert BrowserState is not None
    assert callable(configure_environment_factory)
    assert callable(run)
    assert callable(run_agent)
