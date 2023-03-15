# --------------------------------------------------------------------------
#                         Copyright © by
#           Ignitarium Technology Solutions Pvt. Ltd.
#                         All rights reserved.
#  This file contains confidential information that is proprietary to
#  Ignitarium Technology Solutions Pvt. Ltd. Distribution,
#  disclosure or reproduction of this file in part or whole is strictly
#  prohibited without prior written consent from Ignitarium.
# --------------------------------------------------------------------------
#  Filename    : setup.py
# --------------------------------------------------------------------------
#  Description :
# --------------------------------------------------------------------------
"""Configuration to be set while converting existing fucionalities into python package."""

import setuptools

with open("requirements.txt", encoding="utf-8", mode="r") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="classifier",
    version="0.0.1",
    author="Ignitarium",
    author_email="me@example.com",
    description="Ign classifier pytorch package",
    install_requires=required,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
)
