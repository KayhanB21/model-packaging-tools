# coding=utf-8
# Keyhan B
# V0.1
# Oct 2021
# Required libraries

from dynaconf import Dynaconf, Validator

print("App is started.")
settings = Dynaconf(settings_files = ["config/default_settings.yml",  # a file for default settings
                                      "config/settings.yml",  # a file for main settings
                                      "config/.secrets.yml"  # a file for sensitive data (gitignored)
                                      ],
                    environments = True,  # Enable layered environments
                    env_switcher = "ENV_APP",  # to switch environments
                    )

# settings.validators.register(Validator("NAME", must_exist = True, eq = "Bruno", env = "development"))
# settings.validators.validate()
print(f"Reading the config file is finished.")
