"""DKTM package initializer.

This file marks the ``dktm`` directory as a Python package and
aggregates commonly used modules for convenience.  Importing this
package will expose the high‑level APIs needed to orchestrate the
Dynamic Kernel Transition Mechanism.
"""

# Re‑export key modules for convenience
from . import adapter as adapter  # noqa: F401
from . import command_dict as command_dict  # noqa: F401
from . import config as config  # noqa: F401
from .executor import Executor  # noqa: F401
from . import platform_ops as platform_ops  # noqa: F401