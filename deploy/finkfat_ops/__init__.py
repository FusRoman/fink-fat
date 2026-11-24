"""ff-ops: management tool for the fink-fat production instance on OpenStack.

The package is split into pure modules (configuration models, naming, template and
command builders) and a thin I/O layer (:mod:`finkfat_ops.runner`) that is the only
place where external commands are actually executed. See ``deploy/README.md``.
"""

__version__ = "0.1.0"
