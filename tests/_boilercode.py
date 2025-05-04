# _boilercode.py

"""
For internal consumption during development process, do not run this file.
"""

import pytest


# 1. PARAMETRIZATION
@pytest.mark.parametrize("x, y, expected", [
    (2, 3, 5),
    (1, -1, 0),
    (0, 0, 0),
])
def test_add(x, y, expected):
    assert x + y == expected

# 2. FIXTURE
@pytest.fixture
def sample_user():
    return {"name": "Alice", "age": 30}

def test_user_fixture(sample_user):
    assert sample_user["name"] == "Alice"
    assert sample_user["age"] == 30

# 3. SKIPPING TESTS
def test_skip_example():
    pytest.skip("Skipping this test temporarily")

# 4. EXPECTING EXCEPTIONS
def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        1 / 0

# 5. EXPECTED FAILURE
@pytest.mark.xfail(reason="Not implemented yet")
def test_not_implemented():
    assert False

# 6. CUSTOM ASSERT
def test_custom_assert():
    a = 5
    b = 10
    assert a + b == 15, "Addition result should be 15"

# 7. FIXTURE WITH TEARDOWN
@pytest.fixture
def open_file(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello world")
    try:
        yield file_path
    finally:
        if file_path.exists():
            file_path.unlink()  # teardown: delete the file

def test_file_content(open_file):
    content = open_file.read_text()
    assert content == "hello world"
    assert open_file.exists()

if __name__ == "__main__":
    pytest.main()
