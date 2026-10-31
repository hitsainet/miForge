"""Performance cases (ADR-017's layout, ADR-024's rule). All ``gpu``-marked.

ADR-024: **measure the path you changed.** A number taken from a path a change did not
touch is not a verified fix, and a host-to-device figure taken from an x16 number is wrong
by 3-6x on this node, where **both PCIe links are x4**.
"""
