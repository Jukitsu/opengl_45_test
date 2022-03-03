import pyglet
import ctypes
import glm
from collections import deque
import math
from random import randint

pyglet.options["shadow_window"] = False
pyglet.options["debug_gl"] = False

from OpenGL.GL import *
from OpenGL.GL import shaders
config = pyglet.gl.Config(double_buffer = True,
				major_version = 4, minor_version = 6, depth_size = 16)
window = pyglet.window.Window(config = config, vsync = False, width = 852, height = 480, resizable = True)
fps_display = pyglet.window.FPSDisplay(window)

dim = [852, 480]


quad = [
    -1.0,  1.0,  0.0, 1.0,
    -1.0, -1.0,  0.0, 0.0,
     1.0, -1.0,  1.0, 0.0,

    -1.0,  1.0,  0.0, 1.0,
     1.0, -1.0,  1.0, 0.0,
     1.0,  1.0,  1.0, 1.0
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
     0.5,  0.5,  0.5,   0.0, 1.0, 0.0,    1.0,  0.0,  0.0,
     0.5, -0.5,  0.5,   0.0, 0.0, 0.0,    1.0,  0.0,  0.0,
     0.5, -0.5, -0.5,   1.0, 0.0, 0.0,    1.0,  0.0,  0.0,
     0.5,  0.5, -0.5,   1.0, 1.0, 0.0,    1.0,  0.0,  0.0,

    -0.5,  0.5, -0.5,   0.0, 1.0, 0.0,   -1.0,  0.0,  0.0,
    -0.5, -0.5, -0.5,   0.0, 0.0, 0.0,   -1.0,  0.0,  0.0,
    -0.5, -0.5,  0.5,   1.0, 0.0, 0.0,   -1.0,  0.0,  0.0,
    -0.5,  0.5,  0.5,   1.0, 1.0, 0.0,   -1.0,  0.0,  0.0,

     0.5,  0.5,  0.5,   0.0, 1.0, 0.0,    0.0,  1.0,  0.0,
     0.5,  0.5, -0.5,   0.0, 0.0, 0.0,    0.0,  1.0,  0.0,
    -0.5,  0.5, -0.5,   1.0, 0.0, 0.0,    0.0,  1.0,  0.0,
    -0.5,  0.5,  0.5,   1.0, 1.0, 0.0,    0.0,  1.0,  0.0,

    -0.5, -0.5,  0.5,   0.0, 1.0, 0.0,    0.0, -1.0,  0.0,
    -0.5, -0.5, -0.5,   0.0, 0.0, 0.0,    0.0, -1.0,  0.0,
     0.5, -0.5, -0.5,   1.0, 0.0, 0.0,    0.0, -1.0,  0.0,
     0.5, -0.5,  0.5,   1.0, 1.0, 0.0,    0.0, -1.0,  0.0,

    -0.5,  0.5,  0.5,   0.0, 1.0, 0.0,    0.0,  0.0,  1.0,
    -0.5, -0.5,  0.5,   0.0, 0.0, 0.0,    0.0,  0.0,  1.0,
     0.5, -0.5,  0.5,   1.0, 0.0, 0.0,    0.0,  0.0,  1.0, 
     0.5,  0.5,  0.5,   1.0, 1.0, 0.0,    0.0,  0.0,  1.0,

     0.5,  0.5, -0.5,   0.0, 1.0, 0.0,    0.0,  0.0, -1.0,
     0.5, -0.5, -0.5,   0.0, 0.0, 0.0,    0.0,  0.0, -1.0,
    -0.5, -0.5, -0.5,   1.0, 0.0, 0.0,    0.0,  0.0, -1.0,
    -0.5,  0.5, -0.5,   1.0, 1.0, 0.0,    0.0,  0.0, -1.0
]

instanced_array = []
for i in range(-128, 128):
    for k in range(-128, 128):
        instanced_array.extend((i, randint(-10, 10) / 10, k))

for _ in range(256):
    instanced_array.extend(((512 - randint(0, 1024)) / 8, (512 - randint(0, 1024)) / 8, (512 - randint(0, 1024)) / 8))

light_pos = (-1.4, 3.5, 2.5)
instanced_array.extend(light_pos)

cmd = [
    # Index Count          Instance Count            Base Index   Base Vertex   Base Instance
    len(indices),   len(instanced_array) // 3 - 1,       0,           0,              0, 
    len(indices),                 1,                     0,           0,   len(instanced_array) // 3 - 1
]

vert = """
#version 460 core

layout(location = 0) in vec3 a_VertexPosition;
layout(location = 1) in vec3 a_VertexTexCoords;
layout(location = 2) in vec3 a_VertexNormal;
layout(location = 3) in vec3 a_InstanceOffset;

layout(std140, binding = 0) uniform u_Camera {
    mat4 u_ModelViewProj;
    vec3 u_CameraPos;
};

out vec3 v_Position;
out vec3 v_TexCoords;
out flat vec3 v_Normal;
out flat vec3 v_BaseColor; // for light blocks, using push constants

const vec3 c_BaseColors[2] = vec3[2](
    vec3(0.0f, 0.0f, 0.0f),
    vec3(1.0f, 1.0f, 1.0f)
);

void main(void) {
    v_Position = a_VertexPosition + a_InstanceOffset;
    v_TexCoords = a_VertexTexCoords;
    v_Normal = a_VertexNormal;
    v_BaseColor = c_BaseColors[gl_DrawID]; // Push constant
    gl_Position = u_ModelViewProj * vec4(a_VertexPosition + a_InstanceOffset, 1.0);
}
"""

frag = """
#version 460 core

in vec3 v_Position;
in vec3 v_TexCoords;
in flat vec3 v_Normal;
in flat vec3 v_BaseColor;

layout(location = 0) uniform sampler2DArray u_TextureArraySampler;
layout(std140, binding = 0) uniform u_Camera {
    mat4 u_ModelViewProj;
    vec3 u_CameraPos;
};
layout(std140, binding = 1) uniform u_Lights {
    vec3 u_LightPos;
    float u_AmbientLight;
    float u_SpecularStrength;
};

out vec4 fragColor;

void main(void) {
    float ambientLight = u_AmbientLight;
    vec3 normal = normalize(v_Normal);
    vec3 lightRay = normalize(u_LightPos - v_Position);
    float diffuseLight = max(dot(normal, lightRay), 0.0f);

    vec3 viewRay = normalize(u_CameraPos - v_Position);
    vec3 reflectRay = reflect(-lightRay, normal);
    float specularLight = u_SpecularStrength * pow(max(dot(viewRay, reflectRay), 0.0f), 32);
    
    fragColor = vec4(v_BaseColor, 1.0f) + (diffuseLight + ambientLight + specularLight) * texture(u_TextureArraySampler, v_TexCoords);
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
mat_ubo = GLuint(0)
light_ubo = GLuint(0)
mapped_ubo = [None, None]
fences = deque()

camera_position = glm.vec3(0, 0, 3)
camera_rotation = [-math.tau / 4, 0.0]

user_input = [0, 0, 0]
mouse_captured = [True]

def update_position(delta_time):
    multiplier = 7 * delta_time

    camera_position[1] += user_input[1] * multiplier

    if user_input[0] or user_input[2]:
        angle = camera_rotation[0] - math.atan2(user_input[2], user_input[0]) + math.tau / 4

        camera_position[0] += math.cos(angle) * multiplier
        camera_position[2] += math.sin(angle) * multiplier


@window.event
def on_mouse_motion(x, y, delta_x, delta_y):
    if mouse_captured[0]:

        camera_rotation[0] += delta_x * 0.004
        camera_rotation[1] += delta_y * 0.004

        camera_rotation[1] = max(-math.tau / 4, min(math.tau / 4, camera_rotation[1]))
@window.event
def on_mouse_drag(x, y, delta_x, delta_y, buttons, modifiers):
		on_mouse_motion(x, y, delta_x, delta_y)
	
@window.event
def on_key_press(key, modifiers):
    if not mouse_captured[0]:
        return

    if   key == pyglet.window.key.D: user_input[0] += 1
    elif key == pyglet.window.key.A: user_input[0] -= 1
    elif key == pyglet.window.key.W: user_input[2] += 1
    elif key == pyglet.window.key.S: user_input[2] -= 1

    elif key == pyglet.window.key.SPACE : user_input[1] += 1
    elif key == pyglet.window.key.LSHIFT: user_input[1] -= 1

@window.event
def on_key_release(key, modifiers):
    if not mouse_captured[0]:
        return

    if   key == pyglet.window.key.D: user_input[0] -= 1
    elif key == pyglet.window.key.A: user_input[0] += 1
    elif key == pyglet.window.key.W: user_input[2] -= 1
    elif key == pyglet.window.key.S: user_input[2] += 1

    elif key == pyglet.window.key.SPACE : user_input[1] -= 1
    elif key == pyglet.window.key.LSHIFT: user_input[1] += 1

@window.event
def on_mouse_press(x, y, button, modifiers):
    mouse_captured[0] = not mouse_captured[0]
    window.set_exclusive_mouse(mouse_captured[0])

    return

t = [0]

def tick(delta_time):
    t[0] += delta_time

pyglet.clock.schedule(tick)

def init_all():

    glCreateVertexArrays(1, vao)


    glCreateBuffers(1, vbo)
    glNamedBufferStorage(
        vbo, 
        ctypes.sizeof(GLfloat) * len(vertices), 
        (GLfloat * len(vertices))(*vertices), 
        0
    )

    glVertexArrayVertexBuffer(vao, 0, vbo, 0, 9 * ctypes.sizeof(GLfloat))

    glEnableVertexArrayAttrib(vao, 0)
    glVertexArrayAttribBinding(vao, 0, 0)
    glVertexArrayAttribFormat(vao, 0, 3, GL_FLOAT, GL_FALSE, 0)

    glEnableVertexArrayAttrib(vao, 1)
    glVertexArrayAttribBinding(vao, 1, 0)
    glVertexArrayAttribFormat(vao, 1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat))

    glEnableVertexArrayAttrib(vao, 2)
    glVertexArrayAttribBinding(vao, 2, 0)
    glVertexArrayAttribFormat(vao, 2, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat))


    glCreateBuffers(1, instance_vbo)
    glNamedBufferStorage(
        instance_vbo, 
        ctypes.sizeof(GLfloat) * len(instanced_array), 
        (GLfloat * len(instanced_array))(*instanced_array), 
        0
    )

    glVertexArrayVertexBuffer(vao, 1, instance_vbo, 0, 3 * sizeof(GLfloat))
    glVertexArrayBindingDivisor(vao, 1, 1)

    glVertexArrayAttribBinding(vao, 3, 1)
    glEnableVertexArrayAttrib(vao, 3)
    glVertexArrayAttribFormat(vao, 3, 3, GL_FLOAT, GL_FALSE, 0)


    glCreateBuffers(1, ibo)
    glNamedBufferStorage(ibo, 
        ctypes.sizeof(GLuint) * len(indices), 
        (GLuint * len(indices))(*indices),
        0
    )

    glVertexArrayElementBuffer(vao, ibo)


    glCreateBuffers(1, icbo)
    glNamedBufferStorage(
        icbo,
        ctypes.sizeof(GLuint * len(cmd)),
        (GLuint * len(cmd)) (*cmd),
        0
    )

    glCreateBuffers(1, mat_ubo)
    glNamedBufferStorage(
        mat_ubo, 
        19 * ctypes.sizeof(GLfloat),
        None,
        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT 
    )
    mapped_ubo[0] = glMapNamedBufferRange(
        mat_ubo, 
        0, 
        19 * ctypes.sizeof(GLfloat), 
        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_FLUSH_EXPLICIT_BIT
    )
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, mat_ubo)
    

    light_data = [
        *light_pos, # Light Pos
        0.25, # Ambient
        0.5, # Specular
    ]
    glCreateBuffers(1, light_ubo)
    glNamedBufferStorage(
        light_ubo,
        len(light_data) * ctypes.sizeof(GLfloat),
        (GLfloat * len(light_data))(*light_data),
        0
    )
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, light_ubo)


    glCreateTextures(GL_TEXTURE_2D_ARRAY, 1, tao)
    glTextureParameteri(tao, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTextureParameteri(tao, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTextureStorage3D(
        tao, 4, GL_RGBA8, 16, 16, 256
    )
    cobblestone_image = pyglet.image.load(f"textures/stone.png").get_image_data()
    data = cobblestone_image.get_data("RGBA", cobblestone_image.width * 4)
    glTextureSubImage3D(
        tao, 0, 0, 0, 0,
        cobblestone_image.width, cobblestone_image.height, 1, 
        GL_RGBA, GL_UNSIGNED_BYTE, data
    )
    glGenerateTextureMipmap(tao)
    
    glBindTextureUnit(0, tao)
    glProgramUniform1i(pprcs_program, 0, 0)



    glCreateVertexArrays(1, quad_vao)
    
    glCreateBuffers(1, quad_vbo)
    glNamedBufferStorage(quad_vbo, 
        ctypes.sizeof(GLfloat) * len(quad), 
        (GLfloat * len(quad))(*quad), 
        0
    )

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
    glTextureParameteri(cbo, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTextureParameteri(cbo, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

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
    
def update(delta_time):
    if not mouse_captured[0]:
        user_input[0] = 0
        user_input[1] = 0
        user_input[2] = 0

    update_position(delta_time)

def update_matrices():
    p_matrix = glm.perspective(
                glm.radians(90),
                dim[0] / dim[1], 0.1, 500)

    v_matrix = glm.mat4(1.0)
    v_matrix = glm.rotate(v_matrix, camera_rotation[1], -glm.vec3(1.0, 0.0, 0.0))
    v_matrix = glm.rotate(v_matrix, -camera_rotation[0] - math.tau / 4, -glm.vec3(0.0, 1.0, 0.0))
    v_matrix = glm.translate(v_matrix, -camera_position)
    

    vp_matrix = p_matrix * v_matrix

    ctypes.memmove(mapped_ubo[0], glm.value_ptr(vp_matrix), 16 * ctypes.sizeof(GLfloat))
    ctypes.memmove(mapped_ubo[0] + 16 * ctypes.sizeof(GLfloat), glm.value_ptr(camera_position), 3 * ctypes.sizeof(GLfloat))
    glFlushMappedNamedBufferRange(mat_ubo, 0, 19 * ctypes.sizeof(GLfloat))


pyglet.clock.schedule(update)
window.set_exclusive_mouse(True)

@window.event
def on_draw():
    update_matrices()

    while len(fences) > 3:
        fence = fences.popleft()
        glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 2147483647)
        glDeleteSync(fence)
        
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glClearColor(0.25, 0.25, 0.25, 0.25)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glUseProgram(program)
    glBindVertexArray(vao)
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, icbo)
    
    glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, None, len(cmd) // 5, 0)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDisable(GL_DEPTH_TEST)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(pprcs_program)
    glBindVertexArray(quad_vao)

    glDrawArrays(GL_TRIANGLES, 0, 6)

    glUseProgram(0)
    glBindVertexArray(0)

    fps_display.draw()

    fences.append(glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0))

def main():
    init_all()
    create_fbo()
    pyglet.app.run()

if __name__ == "__main__":
    main()
