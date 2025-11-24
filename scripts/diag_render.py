"""Minimal ModernGL + QOpenGLWidget demo matching provided logic."""

from __future__ import annotations

import sys

import moderngl as GL
import numpy as np
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication


class DebugWidget(QOpenGLWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setOption(QSurfaceFormat.FormatOption.DebugContext)
        self.setFormat(fmt)
        self.ctx: GL.Context | None = None
        self.vao = None
        self.program = None

    def initializeGL(self) -> None:  # pragma: no cover
        self.ctx = GL.create_context(require=330)
        self.ctx.clear(0.0, 0.0, 0.0)
        vertex_shader_code = """
        #version 330 core
        in vec3 in_vert;
        void main() {
            gl_Position = vec4(in_vert, 1.0);
        }
        """
        fragment_shader_code = """
        #version 330 core
        out vec4 fragColor;
        uniform vec4 color;
        void main() {
            fragColor = color;
        }
        """
        self.program = self.ctx.program(
            vertex_shader=vertex_shader_code,
            fragment_shader=fragment_shader_code,
        )
        vertices = np.array(
            [
                0.0,
                0.5,
                0.0,
                -0.5,
                -0.5,
                0.0,
                0.5,
                -0.5,
                0.0,
            ],
            dtype="f4",
        )
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, "in_vert")
        self.program["color"].value = (1.0, 1.0, 1.0, 1.0)

    def resizeGL(self, w: int, h: int) -> None:  # pragma: no cover
        if self.ctx is None:
            return
        fbo = self.ctx.detect_framebuffer()
        fbo.use()
        self.ctx.viewport = (0, 0, w, h)

    def paintGL(self) -> None:  # pragma: no cover
        if self.ctx is None or self.vao is None:
            return
        self.ctx.clear(0.0, 0.0, 0.0)
        self.vao.render()


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    win = DebugWidget()
    win.resize(800, 600)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

