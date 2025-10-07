from ._version import __version__
# Import C++ bindings module built by pybind11_add_module (swift_td)
from swift_sarsa import SwiftSarsa

__all__ = ["SwiftSarsa", "__version__"]
