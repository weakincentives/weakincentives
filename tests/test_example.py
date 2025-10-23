from weakincentives import hello


def test_example():
    """A dummy test to verify pytest is working."""
    assert 1 + 1 == 2


def test_hello():
    """Test the hello function."""
    assert hello() == "Hello from weakincentives!"
