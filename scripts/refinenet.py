#!/usr/bin/env python3

import os
import sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..')))
from refinenet import run_from_args

run_from_args()
