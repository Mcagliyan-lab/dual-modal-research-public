import pytest
import sys

# Add the project root to the sys.path to allow imports from nn_neuroimaging
# This assumes the script is run from the project root or adjusted path.
# If run from C:\source\dual-modal-research, this line ensures src/ is findable.
sys.path.insert(0, "./src")
sys.path.insert(0, "./tests")


if __name__ == "__main__":
    # Run all tests in the tests/ directory with verbose output
    exit_code = pytest.main(["-v", "tests/"])
    sys.exit(exit_code) 