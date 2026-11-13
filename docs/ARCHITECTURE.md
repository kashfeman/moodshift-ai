# Application architecture

## Implemented in Milestone 1

The installable `moodshift` package lives under `src/`. Tests import the installed
package rather than relying on repository-root imports. `config.py` owns immutable,
validated settings. `presets.py` owns immutable descriptive metadata and explicit
ID lookup. Both use only the Python standard library.

The `imaging`, `engine`, `services`, and `ui` packages currently contain only
module documentation. They establish boundaries without introducing stub
processors, fake inference, or speculative interfaces.

## Planned responsibilities

| Module | Responsibility | Planned dependencies |
| --- | --- | --- |
| `config` | Limits and application defaults | Standard library |
| `presets` | Preset identity and descriptive metadata | Standard library |
| `imaging` | Decode, validate, normalize, classical DIP, encode | Pillow, NumPy, OpenCV |
| `engine` | Neural network definitions, weight loading, inference | PyTorch, TorchVision |
| `services` | Coordinate application use cases and resolve presets | Config, presets, imaging, engine |
| `ui` | Input collection, event handling, results and error presentation | Gradio, services |

Planned request flow: UI → service → input validation and decoding → processing
or inference → output encoding → UI. The UI calls services; it does not contain
processing or neural network logic. Imaging and engine code must not import the
UI or application services. Config and preset metadata remain independent of
heavy libraries. Add interfaces only when concrete integration needs justify them.

Future image validation must check encoded bytes before decoding, then decoded
dimensions and pixel count before expensive processing. Upload formats, color
handling, model preprocessing, and error contracts will be decided alongside
their implementations, not simulated in this foundation.

## Assets and validation

`assets/presets/` is reserved for tracked preset resources. Generated images belong
in ignored `outputs/`; weights belong in ignored `weights/`. No weights are bundled
or downloaded. No paid API is part of the design.

Unit tests cover setting constraints, immutable defaults, catalog integrity,
metadata validation, and lookup failures. CI installs the package on Python 3.11,
runs Ruff lint and formatting checks, then runs Pytest. Future milestones should
add processing and inference tests when those implementations exist.
