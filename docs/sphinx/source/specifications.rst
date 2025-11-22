Temper Library Specifications
=============================

This document details the mandatory technical specifications and design
constraints for the Temper library. It serves as the reference for
standardized library behaviours ensuring consistency, predictability, and
robust error handling across all public and internal APIs.

1. Nan Behaviour: Error Prevention
----------------------------------

Description
^^^^^^^^^^^
NaN (Not a Number) values **must** never be allowed to participate in any
tensor operation that would utilize that NaN in a calculation.

Rationale
^^^^^^^^^
NaN indicates an invalid or undefined numeric condition. Allowing it to
propagate across calculations is highly undesirable as it invalidates
results without necessarily raising an immediate, clear error, thereby
serving no purpose in actual computations. **NaNs must always be handled
proactively.**

Specification (Must)
^^^^^^^^^^^^^^^^^^^^

.. admonition:: Mandatory NaN Validation
   :class: primary

   * **Exception Handling:** Operations using tensor values for calculations
     **must** always check for the presence of NaN in the input tensors.
     If a NaN is detected, the operation **must** immediately throw a
     ``"nan_error"`` exception.

   * **Pre-Calculation Check:** This check **must always** be performed
     **before** any calculations using the tensor values are executed in
     any function (including public APIs and internal helper functions).