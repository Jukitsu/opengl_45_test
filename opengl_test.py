import pyglet
import ctypes
import glm

pyglet.options["shadow_window"] = False
pyglet.options["debug_gl"] = False

from OpenGL.GL import *
from OpenGL.GL import shaders
config = pyglet.gl.Config(double_buffer = True,
				major_version = 4, minor_version = 6, depth_size = 16)
window = pyglet.window.Window(config = config, vsync = False, width = 852, height = 480, resizable = True)
fps_display = pyglet.window.FPSDisplay(window)

dim = [852, 480]

vertex_positions = [
	 0.5,  0.5,  0.5,  0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.5,  0.5, -0.5,
	-0.5,  0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,  0.5, -0.5,  0.5,  0.5,
	-0.5,  0.5,  0.5, -0.5,  0.5, -0.5,  0.5,  0.5, -0.5,  0.5,  0.5,  0.5,
	-0.5, -0.5,  0.5, -0.5, -0.5, -0.5,  0.5, -0.5, -0.5,  0.5, -0.5,  0.5,
	-0.5,  0.5,  0.5, -0.5, -0.5,  0.5,  0.5, -0.5,  0.5,  0.5,  0.5,  0.5,
	 0.5,  0.5, -0.5,  0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,  0.5, -0.5,
]

tex_coords = [
	0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
	0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
	0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
	0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
	0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
	0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
]

quad = [
    -1.0,  1.0,  0.0, 1.0,
    -1.0, -1.0,  0.0, 0.0,
     1.0, -1.0,  1.0, 0.0,

    -1.0,  1.0,  0.0, 1.0,
     1.0, -1.0,  1.0, 0.0,
     1.0,  1.0,  1.0, 1.0
]

shading = [
	0.80, 0.80, 0.80, 0.80,
	0.80, 0.80, 0.80, 0.80,
	1.00, 1.00, 1.00, 1.00,
	0.49, 0.49, 0.49, 0.49,
	0.92, 0.92, 0.92, 0.92,
	0.92, 0.92, 0.92, 0.92,
]

indices = [
	 0,  1,  2,  0,  2,  3, # right
	 4,  5,  6,  4,  6,  7, # left
	 8,  9, 10,  8, 10, 11, # top
	12, 13, 14, 12, 14, 15, # bottom
	16, 17, 18, 16, 18, 19, # front
	20, 21, 22, 20, 22, 23, # back
]

vertices = [
     0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
     0.5, -0.5,  0.5,   0.0, 0.0, 0.0,
     0.5, -0.5, -0.5,   1.0, 0.0, 0.0,
     0.5,  0.5, -0.5,   1.0, 1.0, 0.0,

    -0.5,  0.5, -0.5,   0.0, 1.0, 0.0,
    -0.5, -0.5, -0.5,   0.0, 0.0, 0.0,
    -0.5, -0.5,  0.5,   1.0, 0.0, 0.0,
    -0.5,  0.5,  0.5,   1.0, 1.0, 0.0,

    -0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
    -0.5,  0.5, -0.5,   0.0, 0.0, 0.0,
     0.5,  0.5, -0.5,   1.0, 0.0, 0.0,
     0.5,  0.5,  0.5,   1.0, 1.0, 0.0,

    -0.5, -0.5,  0.5,   0.0, 1.0, 0.0,
    -0.5, -0.5, -0.5,   0.0, 0.0, 0.0,
     0.5, -0.5, -0.5,   1.0, 0.0, 0.0,
     0.5, -0.5,  0.5,   1.0, 1.0, 0.0,

    -0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
    -0.5, -0.5,  0.5,   0.0, 0.0, 0.0,
     0.5, -0.5,  0.5,   1.0, 0.0, 0.0,
     0.5,  0.5,  0.5,   1.0, 1.0, 0.0,

     0.5,  0.5, -0.5,   0.0, 1.0, 0.0,
     0.5, -0.5, -0.5,   0.0, 0.0, 0.0,
    -0.5, -0.5, -0.5,   1.0, 0.0, 0.0,
    -0.5,  0.5, -0.5,   1.0, 1.0, 0.0
]

instanced_array = []
for _x in range(-8, 8):
    for _y in range(-8, 8):
        for _z in range(-8, 8):
            instanced_array.extend((_x * 2.0, _y * 2.0, _z * 2.0))

cmd = [
    # Index Count          Instance Count            Base Index   Base Vertex   Base Instance
    len(indices),   len(instanced_array) // 3,            0,           0,              0,     
]

vert = """
#version 460 core

layout(location = 0) in vec3 a_VertexPosition;
layout(location = 1) in vec3 a_VertexTexCoords;
layout(location = 2) in vec3 a_InstanceOffset;

layout(location = 0) uniform mat4 u_ModelViewProj;

out vec3 v_TexCoords;

void main(void) {
    v_TexCoords = a_VertexTexCoords;
    gl_Position = u_ModelViewProj * vec4(a_VertexPosition + a_InstanceOffset, 1.0);
}
"""

frag = """
#version 460 core

in vec3 v_TexCoords;

layout(location = 1) uniform sampler2DArray u_TextureArraySampler;

out vec4 fragColor;

void main(void) {
    fragColor = texture(u_TextureArraySampler, v_TexCoords);
}
"""

ppvert = """
#version 460 core

layout(location = 0) in vec2 a_Position;
layout(location = 1) in vec2 a_TexCoords;

out vec2 v_Position;
out vec2 v_TexCoords;

void main(void) {
    v_Position = a_Position;
    v_TexCoords = a_TexCoords;
    gl_Position = vec4(a_Position, 1.0, 1.0);
}
"""
ppfrag = """
#version 460 core

in vec2 v_Position;
in vec2 v_TexCoords;

layout(location = 0) uniform sampler2D u_ColorBufferSampler;

out vec4 fragColor;

void main(void) {
    fragColor = texture(u_ColorBufferSampler, v_TexCoords) * smoothstep(0.8, 0.0, 0.5 * distance(vec2(0.0), v_Position));
}
"""
vsh = shaders.compileShader(vert, GL_VERTEX_SHADER)
fsh = shaders.compileShader(frag, GL_FRAGMENT_SHADER)

program = shaders.compileProgram(vsh, fsh, validate=True)

ppvsh = shaders.compileShader(ppvert, GL_VERTEX_SHADER)
ppfsh = shaders.compileShader(ppfrag, GL_FRAGMENT_SHADER)
pprcs_program = shaders.compileProgram(ppvsh, ppfsh, validate=True)

vao = GLuint(0)
vbo = GLuint(0)
instance_vbo = GLuint(0)
ibo = GLuint(0)
icbo = GLuint(0)
tao = GLuint(0)
fbo = GLuint(0)
cbo = GLuint(0)
dbo = GLuint(0)
quad_vao = GLuint(0)
quad_vbo = GLuint(0)
ubo = GLuint(0)


t = [0]

def tick(delta_time):
    t[0] += delta_time

pyglet.clock.schedule_interval(tick, 1 / 60)

def init_all():

    glCreateVertexArrays(1, vao)


    glCreateBuffers(1, vbo)
    glNamedBufferStorage(
        vbo, 
        ctypes.sizeof(GLfloat) * len(vertices), 
        (GLfloat * len(vertices))(*vertices), 
        0
    )

    glVertexArrayVertexBuffer(vao, 0, vbo, 0, 6 * ctypes.sizeof(GLfloat))

    glEnableVertexArrayAttrib(vao, 0)
    glVertexArrayAttribBinding(vao, 0, 0)
    glVertexArrayAttribFormat(vao, 0, 3, GL_FLOAT, GL_FALSE, 0)

    glEnableVertexArrayAttrib(vao, 1)
    glVertexArrayAttribBinding(vao, 1, 0)
    glVertexArrayAttribFormat(vao, 1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat))


    glCreateBuffers(1, instance_vbo)
    glNamedBufferStorage(
        instance_vbo, 
        ctypes.sizeof(GLfloat) * len(instanced_array), 
        (GLfloat * len(instanced_array))(*instanced_array), 
        0
    )

    glVertexArrayVertexBuffer(vao, 1, instance_vbo, 0, 3 * sizeof(GLfloat))

    glEnableVertexArrayAttrib(vao, 2)
    glVertexArrayBindingDivisor(vao, 1, 1)
    glVertexArrayAttribFormat(vao, 2, 3, GL_FLOAT, GL_FALSE, 0)
    glVertexArrayAttribBinding(vao, 2, 1)


    glCreateBuffers(1, ibo)
    glNamedBufferStorage(ibo, ctypes.sizeof(GLuint) * len(indices), (GLuint * len(indices))(*indices), 0)

    glVertexArrayElementBuffer(vao, ibo)


    glCreateBuffers(1, icbo)
    glNamedBufferStorage(
        icbo,
        ctypes.sizeof(GLuint * len(cmd)),
        (GLuint * len(cmd)) (*cmd),
        0
    )


    glCreateTextures(GL_TEXTURE_2D_ARRAY, 1, tao)
    glTextureParameteri(tao, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTextureParameteri(tao, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTextureStorage3D(
        tao, 4, GL_RGBA8, 16, 16, 256
    )
    cobblestone_image = pyglet.image.load(f"textures/cobblestone.png").get_image_data()
    data = cobblestone_image.get_data("RGBA", cobblestone_image.width * 4)
    glTextureSubImage3D(
        tao, 0, 0, 0, 0,
        cobblestone_image.width, cobblestone_image.height, 1, 
        GL_RGBA, GL_UNSIGNED_BYTE, data
    )
    glGenerateTextureMipmap(tao)
    
    glBindTextureUnit(0, tao)
    glProgramUniform1i(program, 1, 0)

    glCreateVertexArrays(1, quad_vao)
    
    glCreateBuffers(1, quad_vbo)
    glNamedBufferStorage(quad_vbo, ctypes.sizeof(GLfloat) * len(quad), (GLfloat * len(quad))(*quad), 0)

    glVertexArrayVertexBuffer(quad_vao, 0, quad_vbo, 0, 4 * sizeof(GLfloat))

    glEnableVertexArrayAttrib(quad_vao, 0)
    glVertexArrayAttribBinding(quad_vao, 0, 0)
    glVertexArrayAttribFormat(quad_vao, 0, 2, GL_FLOAT, GL_FALSE, 0)

    glEnableVertexArrayAttrib(quad_vao, 1)
    glVertexArrayAttribBinding(quad_vao, 1, 0)
    glVertexArrayAttribFormat(quad_vao, 1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat))

def create_fbo():
    glCreateFramebuffers(1, fbo)

    glCreateTextures(GL_TEXTURE_2D, 1, cbo)
    glTextureStorage2D(cbo, 1, GL_RGBA8, dim[0], dim[1])
    glTextureParameteri(cbo, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTextureParameteri(cbo, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, cbo, 0)

    glCreateRenderbuffers(1, dbo)
    glNamedRenderbufferStorage(dbo, GL_DEPTH_COMPONENT16, dim[0], dim[1])

    glNamedFramebufferRenderbuffer(fbo, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, dbo)

    assert glCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

    glBindTextureUnit(1, cbo)
    glProgramUniform1i(pprcs_program, 0, 1)

@window.event
def on_resize(width, height):
    glDeleteFramebuffers(1, fbo)
    glDeleteTextures(1, cbo)
    glDeleteRenderbuffers(1, dbo)

    dim[0] = width
    dim[1] = height

    create_fbo()

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glViewport(0, 0, width, height)

def update_matrices(x, y, z):
    p_matrix = glm.perspective(
                glm.radians(90),
                dim[0] / dim[1], 0.1, 500)

    v_matrix = glm.mat4(1.0)
    v_matrix = glm.translate(v_matrix, glm.vec3(-x, -y, -z))
    v_matrix = glm.rotate(v_matrix, glm.sin(t[0] / 3 * 2), -glm.vec3(1.0, 0.0, 0.0))
    v_matrix = glm.rotate(v_matrix, t[0], -glm.vec3(0.0, 1.0, 0.0))

    vp_matrix = p_matrix * v_matrix
    glProgramUniformMatrix4fv(program, 0, 1, GL_FALSE, glm.value_ptr(vp_matrix))

@window.event
def on_draw():
    update_matrices(0, 0, 6)

    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    glUseProgram(program)
    glBindVertexArray(vao)
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, icbo)
    
    glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, None, len(cmd) // 5, 0)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDisable(GL_DEPTH_TEST)
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(pprcs_program)
    glBindVertexArray(quad_vao)

    glDrawArrays(GL_TRIANGLES, 0, 6)

    glUseProgram(0)
    glBindVertexArray(0)

    fps_display.draw()

def main():
    init_all()
    create_fbo()
    pyglet.app.run()

if __name__ == "__main__":
    main()
