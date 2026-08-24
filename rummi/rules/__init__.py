"""The rules as data: configuration, tile encoding, and the action layout.

Backend-free by design. Every implementation under :mod:`rummi.env` derives
its shapes and constants from here, so they cannot drift apart on the parts
that are definitional rather than computational.
"""
