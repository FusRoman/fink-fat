#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

sys.path.insert(0, os.path.abspath(".."))

from time import sleep
from random import random
from random import getrandbits
from clint.textui import progress


if __name__ == "__main__":
    # for i in progress.bar(range(100)):
    #     b = bool(getrandbits(1))

    #     sleep(random() * 0.2)

    bar = progress.Bar(label="fink_fat", expected_size=5)

    sleep(2)
    bar.show(1)
    sleep(1)
    bar.show(2)
    sleep(1)
    bar.show(3)
    sleep(3)
    bar.show(4)
    sleep(2)
    bar.show(5)

    bar.done()

    # for i in progress.dots(range(100)):
    #     sleep(random() * 0.2)

    # for i in progress.mill(range(100)):
    #     sleep(random() * 0.2)

    # # Override the expected_size, for iterables that don't support len()
    # D = dict(zip(range(100), range(100)))
    # for k, v in progress.bar(D.items(), expected_size=len(D)):
    #     sleep(random() * 0.2)
