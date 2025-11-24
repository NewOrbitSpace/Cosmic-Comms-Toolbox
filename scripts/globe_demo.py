"""Standalone Earth-textured globe demo using PySide6 + ModernGL."""

from __future__ import annotations

import math
import sys

import moderngl as GL
import numpy as np
from PIL import Image
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication
from src.ui.constants import EARTH_DAYMAP_FILE


def _build_textured_sphere(segments: int = 64) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    indices: list[int] = []
    for j in range(segments + 1):
        v = j / segments
        theta = math.pi * v
        for i in range(segments + 1):
            u = i / segments
            phi = 2 * math.pi * u
            x = math.sin(theta) * math.cos(phi)
            y = math.sin(theta) * math.sin(phi)
            z = math.cos(theta)
            lon = math.atan2(x, y)
            lat = math.asin(z / max(math.sqrt(x * x + y * y + z * z), 1e-6))
            tex_u = 0.5 - (lon / (2 * math.pi))
            tex_v = 0.5 + (lat / math.pi)
            vertices.append([x, y, z, tex_u, tex_v])
    row = segments + 1
    for j in range(segments):
        for i in range(segments):
            a = j * row + i
            b = a + row
            indices.extend([a, b, a + 1, a + 1, b, b + 1])
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


def _matrix_bytes(mat: np.ndarray) -> bytes:
    return np.asarray(mat, dtype=np.float32).T.tobytes()


class EarthGlobeWidget(QOpenGLWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)
        self.ctx: GL.Context | None = None
        self.vao = None
        self.program = None
        self.texture = None
        self._time = 0.0

    def initializeGL(self) -> None:  # pragma: no cover
        self.ctx = GL.create_context(require=330)
        self.ctx.enable(GL.DEPTH_TEST | GL.CULL_FACE)
        vertices, indices = _build_textured_sphere()
        vbo = self.ctx.buffer(vertices.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        self.program = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 mvp;
                in vec3 in_pos;
                in vec2 in_uv;
                out vec3 v_normal;
                out vec2 v_uv;
                void main() {
                    v_normal = normalize(in_pos);
                    v_uv = vec2(in_uv.x, 1.0 - in_uv.y);
                    gl_Position = mvp * vec4(in_pos, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_normal;
                in vec2 v_uv;
                out vec4 fragColor;
                uniform sampler2D surface;
                void main() {
                    vec3 light_dir = normalize(vec3(0.4, -0.5, 0.7));
                    float diffuse = clamp(dot(normalize(v_normal), light_dir), 0.1, 1.0);
                    vec3 color = texture(surface, v_uv).rgb;
                    fragColor = vec4(color * diffuse, 1.0);
                }
            """,
        )
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (vbo, "3f 2f", "in_pos", "in_uv"),
            ],
            ibo,
            index_element_size=4,
        )
        image = Image.open(EARTH_DAYMAP_FILE).convert("RGB")
        self.texture = self.ctx.texture(image.size, 3, image.tobytes())
        self.texture.build_mipmaps()
        self.texture.filter = (GL.LINEAR_MIPMAP_LINEAR, GL.LINEAR)
        self.startTimer(16)

    def timerEvent(self, event) -> None:  # pragma: no cover
        self._time += 0.01
        self.update()

    def resizeGL(self, w: int, h: int) -> None:  # pragma: no cover
        if self.ctx is None:
            return
        self._bind_default_fbo(w, h)

    def paintGL(self) -> None:  # pragma: no cover
        if self.ctx is None or self.vao is None or self.program is None:
            return
        self._bind_default_fbo(self.width(), self.height())
        aspect = max(self.width(), 1) / max(self.height(), 1)
        proj = self._perspective(math.radians(45.0), aspect, 0.1, 10.0)
        eye = np.array([0.0, -3.0, 1.2], dtype=np.float32)
        view = self._look_at(eye, np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 1.0]))
        model = self._rotation_x(math.radians(-23.5)) @ self._rotation_z(self._time)
        mvp = proj @ view @ model
        self.ctx.clear(0.03, 0.03, 0.06, 1.0)
        self.program["mvp"].write(_matrix_bytes(mvp))
        if self.texture is not None:
            self.texture.use(location=0)
        self.vao.render()

    def _bind_default_fbo(self, w: int, h: int) -> None:
        fbo = self.ctx.detect_framebuffer()
        fbo.use()
        self.ctx.viewport = (0, 0, max(int(w), 1), max(int(h), 1))

    @staticmethod
    def _perspective(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
        f = 1.0 / math.tan(fov / 2.0)
        mat = np.zeros((4, 4), dtype=np.float32)
        mat[0, 0] = f / aspect
        mat[1, 1] = f
        mat[2, 2] = (far + near) / (near - far)
        mat[2, 3] = (2 * far * near) / (near - far)
        mat[3, 2] = -1.0
        return mat

    @staticmethod
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

    @staticmethod
    def _rotation_y(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        mat = np.identity(4, dtype=np.float32)
        mat[0, 0] = c
        mat[0, 2] = s
        mat[2, 0] = -s
        mat[2, 2] = c
        return mat

    @staticmethod
    def _rotation_z(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        mat = np.identity(4, dtype=np.float32)
        mat[0, 0] = c
        mat[0, 1] = -s
        mat[1, 0] = s
        mat[1, 1] = c
        return mat

    @staticmethod
    def _rotation_x(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        mat = np.identity(4, dtype=np.float32)
        mat[1, 1] = c
        mat[1, 2] = -s
        mat[2, 1] = s
        mat[2, 2] = c
        return mat


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    widget = EarthGlobeWidget()
    widget.resize(800, 600)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

