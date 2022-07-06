To run a specific set of tests during development
- prepend the test functions with @pytest.mark.select decorator
- run the selected tests with: pytest -s -m select
