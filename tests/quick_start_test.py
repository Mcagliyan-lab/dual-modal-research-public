import os
import sys
import json
import subprocess
import unittest

class TestQuickStartScript(unittest.TestCase):
    """
    Test case for verifying the functionality of examples/quick_start.py.
    """
    
    def setUp(self):
        """Set up for test: define paths and ensure results directory exists."""
        self.script_path = "examples/quick_start.py"
        self.results_dir = "results"
        self.output_json = os.path.join(self.results_dir, "quick_test_results.json")
        self.output_png = os.path.join(self.results_dir, "quick_test_results.png")

        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

    def tearDown(self):
        """Clean up after test: remove generated files."""
        if os.path.exists(self.output_json):
            os.remove(self.output_json)
        if os.path.exists(self.output_png):
            os.remove(self.output_png)
        # Optionally remove the results directory if empty, but usually not needed
        # if not os.listdir(self.results_dir):
        #     os.rmdir(self.results_dir)

    def test_script_execution_and_output(self):
        """
        Test if quick_start.py runs successfully and produces expected output files.
        """
        print(f"\nRunning script: {self.script_path}")
        try:
            # Run the script as a subprocess
            process = subprocess.run(
                [sys.executable, self.script_path],
                capture_output=True,
                text=True,
                check=True # Raise an exception for non-zero exit codes
            )
            print("Script output (stdout):")
            print(process.stdout)
            if process.stderr:
                print("Script errors (stderr):")
                print(process.stderr)

            # Assert that the script executed successfully
            self.assertEqual(process.returncode, 0, f"Script exited with non-zero status: {process.returncode}")

            # Assert that the output files were created
            self.assertTrue(os.path.exists(self.output_json), f"JSON output file not found: {self.output_json}")
            self.assertTrue(os.path.exists(self.output_png), f"PNG output file not found: {self.output_png}")

            # Validate the content of the JSON output
            with open(self.output_json, 'r') as f:
                results = json.load(f)
            
            self.assertIn("test_status", results)
            self.assertEqual(results["test_status"], "SUCCESS", "Test status in JSON is not 'SUCCESS'")
            self.assertIn("operational_state", results)
            self.assertIn("frequency_analysis", results)
            self.assertTrue(len(results["frequency_analysis"]) > 0, "Frequency analysis results are empty")
            
            print("quick_start.py executed successfully and produced valid outputs.")

        except subprocess.CalledProcessError as e:
            self.fail(f"Script execution failed with error: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        except FileNotFoundError:
            self.fail(f"Python interpreter or script not found. Make sure Python is in PATH and script_path is correct.")
        except json.JSONDecodeError:
            self.fail(f"Failed to decode JSON from {self.output_json}. Check script output format.")
        except Exception as e:
            self.fail(f"An unexpected error occurred during test: {e}")

if __name__ == "__main__":
    unittest.main() 