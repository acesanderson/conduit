# test_config.py
def test_config_working():
    print("🔍 This print should be visible if -s is working")
    import logging
    logging.info("🔍 This log should be visible if log_cli is working")
    assert True
