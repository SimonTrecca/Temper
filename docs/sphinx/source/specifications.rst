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
     addressed through a dedicated macro (for example,
     ``TEMPER_CHECK``). This macro must accept a condition, an exception
     type, and an error message.

   * **Macro Behavior:**

     1. **Default:** If the disable flag is **not** set, the macro must
        evaluate the condition. If the condition is true, it must throw
        the specified exception with the provided message.

     2. **Disabled:** If ``TEMPER_DISABLE_ERROR_CHECKS`` **is** set, the
        macro must resolve to a no-op (such as ``((void)0)``), ensuring
        the check and the branch are removed during compilation.

2. Device-Side Error Reporting: Kernel Safety
---------------------------------------------

Description
^^^^^^^^^^^
Standard C++ exceptions cannot be thrown from within SYCL kernels or other
device-side code. A dedicated protocol **must** be used to detect errors
during asynchronous execution and propagate them back to the host.

Rationale
^^^^^^^^^
Device code executes in parallel across many work-items. When an invalid
state is reached (for example, NaN detection), the specific work-item must
cease execution immediately to prevent undefined behavior. Since the host
executes asynchronously, it requires a persistent, thread-safe signal to
determine if the kernel completed successfully or if an exception should be
raised.

Specification (Must)
^^^^^^^^^^^^^^^^^^^^

.. admonition:: Atomic Device-Host Error Protocol
   :class: primary

   * **Atomic Flag:** Functions dispatching kernels with runtime checks
     **must** allocate a shared or global memory integer flag (accessible by
     both device and host) and initialize it to zero before dispatch.

   * **Macro Requirement:** All device-side checks **must** be implemented
     via a dedicated macro (for example, ``TEMPER_DEVICE_CHECK``). This
     macro must accept a boolean condition, the pointer to the error flag,
     and a unique error code.

   * **Kernel Behavior:**

     1. **Evaluation:** The macro must evaluate the condition.

     2. **Atomic Set:** If the condition is true, the macro **must**
        atomically set the error flag to the provided code (for example,
        using ``atomic_ref``).

     3. **Abort:** The macro **must** immediately force a return from the
        current work-item to stop further execution.

   * **Host Translation:** After waiting for kernel completion, the host
     **must** inspect the value of the error flag. If the flag is non-zero,
     the host **must** use the standard ``TEMPER_CHECK`` macro to throw the
     specific C++ exception corresponding to that error code.

   * **Disable Switch:** If ``TEMPER_DISABLE_ERROR_CHECKS`` is defined,
     the device macro **must** resolve to a no-op, removing all atomic
     operations and branching from the compiled kernel.

3. NaN Behaviour: Error Prevention
----------------------------------

Description
^^^^^^^^^^^
NaN (Not a Number) values **must** never be allowed to participate in any
tensor operation that would utilize that NaN in a calculation.

Rationale
^^^^^^^^^
NaN indicates an invalid or undefined numeric condition. Allowing it to
propagate across calculations is highly undesirable as it invalidates
results without necessarily raising an immediate, clear error. NaNs must
always be handled proactively.

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

4. Non-Finite Behaviour: Invalid Result Prevention
--------------------------------------------------

Description
^^^^^^^^^^^
Non-finite values (``+∞``, ``-∞``, or overflow results) must never be
allowed to propagate as valid outputs of any computation within the Temper
library. While NaN represents an invalid *input* condition, non-finite values
represent an invalid or unstable *result* that must be caught immediately
after it is produced.

Rationale
^^^^^^^^^
Infinity or overflow results cannot be meaningfully used in subsequent
computations and typically indicate numerical instability or exceeding the
representable range of the underlying data type. These values must be
detected and reported in a unified and consistent manner across all APIs.

Specification (Must)
^^^^^^^^^^^^^^^^^^^^

.. admonition:: Mandatory Non-Finite Result Validation
   :class: primary

   * **Result Validation:** Any operation that produces new tensor values
     **must** validate that the resulting values are finite. If a non-finite
     value is detected, the operation **must** immediately raise a
     ``"nonfinite_error"`` exception.

   * **Post-Calculation Check:** This validation **must always** occur
     *after* performing a computation that generates new numeric values,
     but **before** returning them to any caller (internal or external).

5. ``validation_error`` Policy: Invalid Inputs
----------------------------------------------

Description
^^^^^^^^^^^
``validation_error`` is the canonical exception for invalid user inputs,
incompatible tensor metadata, or precondition failures detected **before**
computation begins.

Rationale
^^^^^^^^^
Validation failures are deterministic and should be caught early to
prevent wasted device dispatch and undefined behaviour. A single error
type ensures consistency across public APIs and internal helpers.

Specification (Must)
^^^^^^^^^^^^^^^^^^^^

.. admonition:: Validation Error Enforcement
   :class: primary

   * **When to Use:** The library **must** throw ``validation_error`` when:

     1. A public API receives invalid arguments (e.g., null handles).
     2. Tensor metadata is incompatible (e.g., mismatched ranks).
     3. Any precondition is violated **before** kernel dispatch.

   * **Enforcement:** All checks **must** be implemented via the
     centralized error reporting utility and executed **before**
     allocating device error flags or launching kernels.

   * **Message:** The error message **must** identify the parameter/tensor,
     the violated constraint, and the operation context.


6. ``bounds_error`` Policy: Indexing Violations
-----------------------------------------------

Description
^^^^^^^^^^^
``bounds_error`` is the canonical exception for index out-of-range and
invalid bounds conditions during tensor indexing or storage access.

Rationale
^^^^^^^^^
Bounds issues are distinct from general validation as they imply access
outside legal memory ranges. Separating them allows consistent mapping
from device-side error codes to a specific host exception.

Specification (Must)
^^^^^^^^^^^^^^^^^^^^

.. admonition:: Bounds Error Enforcement
   :class: primary

   * **When to Use:** The library **must** throw ``bounds_error`` when:

     1. An index is < 0 or >= dimension extent.
     2. Slice ranges or computed offsets exceed valid storage spans.

   * **Device Mapping:** If bounds checks occur inside kernels, the host
     **must** translate the resulting device error code into
     ``bounds_error``.

   * **Enforcement:** Host checks **must** occur **before** any memory
     access depending on the index. The centralized utility **must** be
     used to raise the exception.


7. ``computation_error`` Policy: General Numerical Failures
-----------------------------------------------------------

Description
^^^^^^^^^^^
``computation_error`` is the canonical exception for numerical failures
occurring **during** or **as a result of** computation that are **not**
covered by specific NaN or non-finite policies.

Rationale
^^^^^^^^^
Distinct exception types for NaN and non-finite results (Specs #3 & #4)
allow users to target specific data issues. ``computation_error`` serves
as the handler for remaining mathematical failures (e.g., divergence).

Specification (Must)
^^^^^^^^^^^^^^^^^^^^

.. admonition:: Computation Error Enforcement
   :class: primary

   * **When to Use:** The library **must** throw ``computation_error`` when:

     1. A computation fails to converge or satisfy numerical post-conditions.
     2. A mathematical domain error occurs that is not strictly an input
        validation issue.

   * **Exclusions:**

     - **NaN Validation** (Spec #3) **must** throw ``nan_error``.
     - **Non-Finite Validation** (Spec #4) **must** throw ``nonfinite_error``.

   * **Enforcement:** Host checks must use the centralized utility. Device
     checks use the kernel error protocol and are translated on host.


8. ``device_error`` Policy: Execution Failures
----------------------------------------------

Description
^^^^^^^^^^^
``device_error`` is the canonical exception for failures related to SYCL
device execution, runtime exceptions, or asynchronous protocol faults.

Rationale
^^^^^^^^^
A dedicated error type is required to distinguish "runtime/environment
failures" from mathematical errors or input validation issues.

Specification (Must)
^^^^^^^^^^^^^^^^^^^^

.. admonition:: Device Error Enforcement
   :class: primary

   * **When to Use:** The library **must** throw ``device_error`` when:

     1. The SYCL runtime reports an exception (sync or async).
     2. The device-host atomic protocol reports a code reserved for runtime
        faults (e.g., illegal state, resource failure).

   * **SYCL Translation:** If SYCL throws a host exception, the library
     **must** catch it at the API boundary and rethrow as ``device_error``,
     preserving the original message.

   * **Device Protocol:** Kernel-level runtime faults **must** use the
     device error protocol and be translated to ``device_error`` on host.