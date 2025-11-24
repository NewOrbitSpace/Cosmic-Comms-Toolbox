"""ModernGL diagnostics: prints context info and matrix samples."""

from __future__ import annotations

import math

import moderngl
import numpy as np
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication
from PySide6.QtOpenGLWidgets import QOpenGLWidget


def _perspective(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fov / 2.0)
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2 * far * near) / (near - far)
    mat[3, 2] = -1.0
    return mat


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    mat = np.identity(4, dtype=np.float32)
    mat[0, :3] = s
    mat[1, :3] = u
    mat[2, :3] = -f
    mat[:3, 3] = -mat[:3, :3] @ eye
    return mat


class _DummyWidget(QOpenGLWidget):
    def initializeGL(self) -> None:  # pragma: no cover
        pass


def main() -> None:
    app = QApplication.instance() or QApplication([])
    widget = _DummyWidget()
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    widget.setFormat(fmt)
    widget.resize(1, 1)
    widget.show()
    widget.hide()
    ctx = moderngl.create_context()
    info = ctx.info
    print("=== ModernGL Context Info ===")
    print("GL_VERSION:", ctx.version_code)
    print("GL_VENDOR:", info.get("GL_VENDOR"))
    print("GL_RENDERER:", info.get("GL_RENDERER"))
    print("GL_MAX_TEXTURE_SIZE:", info.get("GL_MAX_TEXTURE_SIZE"))
    print("GL_MAX_VERTEX_ATTRIBS:", info.get("GL_MAX_VERTEX_ATTRIBS"))
    print("GL_MAX_UNIFORM_BLOCK_SIZE:", info.get("GL_MAX_UNIFORM_BLOCK_SIZE"))
    proj = _perspective(math.radians(45.0), 1.6, 0.1, 10.0)
    view = _look_at(
        np.array([2.0, 2.0, 1.5], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    mvp = proj @ view
    print("\nSample Projection Matrix (row-major):")
    print(np.array2string(proj, precision=3, suppress_small=True))
    print("\nSample View Matrix (row-major):")
    print(np.array2string(view, precision=3, suppress_small=True))
    print("\nSample MVP Matrix (row-major):")
    print(np.array2string(mvp, precision=3, suppress_small=True))


if __name__ == "__main__":
    main()

