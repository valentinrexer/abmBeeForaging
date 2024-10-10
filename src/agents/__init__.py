# Import main classes to make them available at the package level
import mesa

# Define package-level constants
VERSION = "1.0.0"

# Maybe some minimal initialization
print(f"Initializing mypackage version {VERSION}")

# Define what gets imported with "from mypackage import *"
__all__ = ['mesa']