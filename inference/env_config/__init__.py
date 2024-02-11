# env_config.py
import os
import re
from dotenv import load_dotenv


class EnvConfig:
    def __init__(self, template_path='.env.template'):
        self._vars = {}
        load_dotenv()  # Load variables from .env if present
        self._load_template_vars(template_path)
        self._load_environment_vars()
        self._print_vars()

    def _print_vars(self):
        """Print the environment variables and their values."""
        print("Environment variables:")
        for var, value in self._vars.items():
            print(f"{var}={value}")

    def _load_template_vars(self, template_path):
        """Parse the .env.template file to identify required environment variables."""
        try:
            with open(template_path, 'r') as file:
                for line in file:
                    # Skip comment lines and empty lines
                    if line.startswith('#') or not line.strip():
                        continue
                    var_name = line.split('=')[0].strip()
                    self._vars[var_name] = None  # Initialize with None
        except FileNotFoundError:
            print(f"Template file {template_path} not found. Ensure it exists.")
            exit(1)

    def _load_environment_vars(self):
        """Load and set environment variables based on the template."""
        for var in self._vars.keys():
            value = os.getenv(var)
            if value is None:
                print(f"Warning: Environment variable {var} not found or not set.")
            else:
                # Convert the variable name to lowercase snake_case for attribute access
                attr_name = self._convert_to_snake_case(var)
                # remove the original variable from the config
                setattr(self, attr_name, value)

    def _convert_to_snake_case(self, text):
        """Convert uppercase text to snake_case, keeping acronyms intact."""
        # This pattern targets all capital letters that precede lowercase letters (indicating the start of a new word)
        # and all lowercase letters that precede capital letters (indicating the end of a word/start of an acronym).
        # It inserts an underscore between these letters.
        snake_cased = re.sub(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', '_', text)

        # Finally, convert the entire string to lowercase.
        return snake_cased.lower()

    def __getattr__(self, name):
        """Allow direct access to variables as attributes."""
        try:
            return self._vars[name]
        except KeyError:
            raise AttributeError(f"{name} not found in environment configuration.")
