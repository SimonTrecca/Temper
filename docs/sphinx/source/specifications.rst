Temper Library Specifications
=============================

This document details the mandatory technical specifications and design
constraints for the Temper library. It serves as the reference for
standardized library behaviours ensuring consistency, predictability, and
robust error handling across all public and internal APIs.

1. Error Handling: Centralized Management
-----------------------------------------

Description
^^^^^^^^^^^
There needs to be a singular, uniform, and centralized mechanism to handle
error management throughout the library.

Rationale
^^^^^^^^^
Centralization allows for single-point modification of error handling logic.
Furthermore, high-performance applications require the ability to disable
every runtime check to maximize execution speed. A centralized macro ensures
that switching between "safe" and "fast" modes is a simple compile-time
toggle.

Specification (Must)
^^^^^^^^^^^^^^^^^^^^

.. admonition:: Unified Error Enforcement
   :class: primary

   * **Compile-Time Switch:** There shall be a global flag
     ``TEMPER_DISABLE_ERROR_CHECKS``. When defined (e.g., via
     ``-DTEMPER_DISABLE_ERROR_CHECKS=ON``), all runtime checks **must** be
     stripped from the compiled binary to ensure zero overhead.

   * **Kernel Consistency:** All kernel-level runtime checks **must** also
     be capable of being disabled through the usage of this flag.

   * **Macro Definition:** All errors in the Temper library **must** be
     addressed through a dedicated macro (e.g., ``TEMPER_CHECK``). This
     macro must accept a condition, an exception type, and an error message.

   * **Macro Behavior:**

     1. **Default:** If the disable flag is **not** set, the macro must
        evaluate the condition. If the condition is true, it must throw
        the specified exception with the provided message.

     2. **Disabled:** If ``TEMPER_DISABLE_ERROR_CHECKS`` **is** set, the
        macro must resolve to a no-op (such as ``((void)0)``), ensuring
        the check and the branch are removed during compilation.

2. Nan Behaviour: Error Prevention
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

3. Non-Finite Behaviour: Invalid Result Prevention
--------------------------------------------------

Description
^^^^^^^^^^^
Non-finite values (``+∞``, ``-∞``, or overflow results) must never be allowed
to propagate as valid outputs of any computation within the Temper library.
While NaN represents an invalid *input* condition, non-finite values represent
an invalid or unstable *result* that must be caught immediately after it is
produced.

Rationale
^^^^^^^^^
Infinity or overflow results cannot be meaningfully used in subsequent
computations and typically indicate numerical instability or exceeding the
representable range of the underlying data type. These values must be detected
and reported in a unified and consistent manner across all APIs.

Specification (Must)
^^^^^^^^^^^^^^^^^^^^

.. admonition:: Mandatory Non-Finite Result Validation
   :class: primary

   * **Result Validation:**
     Any operation that produces new tensor values **must** validate that
     the resulting values are finite. If a non-finite value is detected,
     the operation **must** immediately raise a ``"nonfinite_error"``
     exception.

   * **Post-Calculation Check:**
     This validation **must always** occur *after* performing a computation
     that generates new numeric values, but **before** returning them to any
     caller (internal or external).

