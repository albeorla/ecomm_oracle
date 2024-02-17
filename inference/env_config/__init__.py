import sys

import json
import os
import re
from dotenv import load_dotenv
from loguru import logger


class EnvConfig:
    def __init__(self, template_path='.env.template'):
        self._vars = {}
        load_dotenv()
        logger.remove()
        logger.add(sys.stdout, level=os.getenv("LOG_LEVEL", "INFO").upper())
        self._load_template_vars(template_path)
        self._load_environment_vars()
        self._print_vars()

    def _print_vars(self):
        """Print the environment variables and their values."""
        logger.debug("Environment variables:")
        config_dict = self._get_snake_cased_config_dict()
        config_json = json.dumps(config_dict, indent=4)
        logger.debug(config_json)

    def _get_snake_cased_config_dict(self):
        """Generate a dictionary with snake-cased attribute names and their values."""
        config_dict = {}
        for var, value in self._vars.items():
            snake_case_var = self._convert_to_snake_case(var)
            config_dict[snake_case_var] = value
        return config_dict

    def _load_template_vars(self, template_path):
        """Parse the .env.template file to identify required environment variables."""
        try:
            with open(template_path, 'r') as file:
                for line in file.readlines():
                    # Skip comment lines and empty lines
                    if line.startswith('#') or not line.strip():
                        continue
                    var_name = line.split('=')[0].strip()
                    self._vars[var_name] = os.getenv(var_name, None)
        except FileNotFoundError:
            logger.debug(f"Template file {template_path} not found. Ensure it exists.")
            exit(1)

    def _load_environment_vars(self):
        """Load and set environment variables based on the template."""
        for var in self._vars.keys():
            value = os.getenv(var)
            if value is None:
                logger.debug(f"Warning: Environment variable {var} not found or not set.")
            else:
                # Convert the variable name to lowercase snake_case for attribute access
                attr_name = self._convert_to_snake_case(var)
                # Convert integer and float strings to their respective types
                if value.isdigit():
                    value = int(value)
                elif '.' in value and value.replace('.', '').isdigit():
                    value = float(value)
                elif value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                setattr(self, attr_name, value)
                self._vars[var] = value

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
